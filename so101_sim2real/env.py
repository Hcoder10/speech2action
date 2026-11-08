"""Gymnasium environment for the owner's SO-101, built so a policy trained here runs unchanged on the bench.

What makes it transfer (each item is something a naive MuJoCo env gets wrong):
  * Same I/O as the real arm: owner-frame degrees + gripper 0..100, 30 Hz, LeRobot clipping and
    `max_relative_target`. The real twin (`real.SO101RealEnv`) has identical spaces.
  * System-identified servo: PD inside a torque clip, back-EMF speed limit, friction, rotor inertia, command
    latency, gripper overload cut-off (`servo.py`), randomised inside the fit's own uncertainty.
  * Real-time loop timing: the observation is read at t_k, the goal reaches the bus after a compute delay, the
    servo acts after the bus latency. Period jitter included.
  * Sensor model: Present_Position quantised to encoder ticks with noise; wrist camera at the bench-fitted mount
    with exposure, white balance, blur, noise and JPEG; jittered mount and FOV.
  * Calibration error: the owner-to-model joint offsets are fitted from images, not measured, so every episode
    perturbs them (lift/elbow split is the least certain).
  * Objects and scene: cube mass/friction/size, table wood texture and tint, lights, table tilt.

Privileged state (cube pose, true joints) is in `info["privileged"]` for asymmetric critics / teachers. It is
never in the observation, because the real arm cannot produce it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import textures
from .contract import CONTROL_HZ, Contract, goal_from_action, load_contract, preprocess_image
from .servo import ParamRanges, ServoParams, load_params
from .sim import ArmSim

TABLE_TOP = 0.0
CUBE_HALF = 0.0285
WOOD_POOL = 16
_WOOD_CACHE: dict = {}


def _wood(i: int, size: int) -> np.ndarray:
    """A pool of procedural table tops (2048^2 takes ~1 s to make), built lazily and shared by every env."""
    if (i, size) not in _WOOD_CACHE:
        palettes = list(textures.WOOD_PALETTES)
        _WOOD_CACHE[(i, size)] = textures.wood(1000 + 17 * i, palettes[i % len(palettes)], size)
    return _WOOD_CACHE[(i, size)]


@dataclass
class DRConfig:
    """Domain randomisation. `scale` multiplies every width (0 = nominal sim, 1 = default, 2 = harder)."""
    scale: float = 1.0
    servo: bool = True
    offset_sigma_deg: tuple = (1.0, 2.5, 2.5, 2.0, 3.0, 1.5)   # calibration error per joint (gripper in units)
    encoder_noise_ticks: float = 0.5
    compute_delay: tuple = (0.004, 0.030)                      # s, read -> write
    period_jitter: float = 0.002                               # s
    cam_pos_mm: float = 3.0
    cam_rot_deg: float = 2.0
    cam_fovy_deg: float = 3.0
    light_scale: tuple = (0.65, 1.2)
    image_gain: tuple = (0.8, 1.15)
    image_wb: float = 0.08
    image_gamma: tuple = (0.85, 1.2)
    image_blur: tuple = (0.0, 1.0)
    image_noise: tuple = (0.005, 0.03)
    jpeg_quality: tuple = (60, 95)
    cube_mass: tuple = (0.04, 0.15)                            # kg (the bench cube is 90 g)
    cube_friction: tuple = (0.35, 1.0)
    cube_size_frac: float = 0.08
    link_mass_frac: float = 0.08
    table_tilt_deg: float = 0.5
    textures: bool = True

    @classmethod
    def off(cls) -> "DRConfig":
        return cls(scale=0.0, servo=False, textures=False)


@dataclass
class TaskSpec:
    name: str = "reach"               # reach | lift | free
    episode_seconds: float = 8.0
    success_radius: float = 0.015     # reach, m
    lift_height: float = 0.05         # lift, m above rest
    extra: dict = field(default_factory=dict)


class SO101Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, task: str | TaskSpec = "reach", obs_keys=("joints", "prev_goal", "target"),
                 action_mode: str = "delta", max_delta: float = 6.0, max_relative_target: float | None = 10.0,
                 params=None, dr: DRConfig | None = None, image_size=(128, 128), camera: str = "wrist",
                 render_mode: str | None = "rgb_array", seed: int | None = None):
        super().__init__()
        self.task = task if isinstance(task, TaskSpec) else TaskSpec(task)
        self.obs_keys = tuple(obs_keys)
        self.action_mode = action_mode
        self.max_delta = float(max_delta)
        self.max_relative_target = max_relative_target
        self.dr = dr if dr is not None else DRConfig()
        self.image_size = tuple(image_size)
        self.camera = camera
        self.render_mode = render_mode
        self.nominal_params, self.ranges = load_params(params)
        self.contract: Contract = load_contract()
        self.sim = ArmSim(self.nominal_params.copy(), self.contract)
        self._nominal = self._snapshot_model()
        self._rng = np.random.default_rng(seed)
        self._tex_dirty = False
        self._lut = None

        lo, hi = (self.contract.low - 15).astype(np.float32), (self.contract.high + 15).astype(np.float32)   # calibration error + noise can read past the limits
        obs = {
            "joints": spaces.Box(lo, hi, dtype=np.float32),
            "prev_goal": spaces.Box(lo, hi, dtype=np.float32),
            "target": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "wrist": spaces.Box(0, 255, shape=(*self.image_size[::-1], 3), dtype=np.uint8),
        }
        self.observation_space = spaces.Dict({k: obs[k] for k in self.obs_keys})
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    # ------------------------------------------------------------------ model snapshot / randomisation
    def _snapshot_model(self) -> dict:
        m = self.sim.model
        return {k: getattr(m, k).copy() for k in ("cam_pos", "cam_quat", "cam_fovy", "light_diffuse", "body_mass",
                                                   "body_inertia", "geom_size", "geom_friction", "geom_rgba", "mat_rgba", "body_gravcomp")} | \
               {"gravity": m.opt.gravity.copy()}

    def _restore_model(self) -> None:
        m = self.sim.model
        for k, v in self._nominal.items():
            if k == "gravity":
                m.opt.gravity[:] = v
            else:
                getattr(m, k)[:] = v

    def _u(self, lo, hi):
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * self.dr.scale
        return float(self._rng.uniform(mid - half, mid + half))

    def _randomise(self) -> None:
        m, s, rng, dr = self.sim.model, self.dr.scale, self._rng, self.dr
        self._restore_model()
        p = self.ranges.sample(self.nominal_params, rng) if (dr.servo and s > 0) else self.nominal_params.copy()
        if dr.servo and s != 1.0 and s > 0:   # rescale the sampled deviation
            for k in ("kp", "kv", "tau", "omega_free", "frictionloss", "damping", "armature"):
                base = getattr(self.nominal_params, k)
                setattr(p, k, base * (getattr(p, k) / base) ** s)
            p.latency = self.nominal_params.latency + (p.latency - self.nominal_params.latency) * s
        self.sim.set_params(p)
        self.ep_params = p
        self.offset_err = rng.normal(0, 1, 6) * np.asarray(dr.offset_sigma_deg) * s
        self.offset_err = np.clip(self.offset_err, -2 * np.asarray(dr.offset_sigma_deg) * s, 2 * np.asarray(dr.offset_sigma_deg) * s)
        # wrist camera mount
        cid = m.camera("wrist").id
        m.cam_pos[cid] += rng.uniform(-1, 1, 3) * dr.cam_pos_mm * 1e-3 * s
        rv = rng.normal(0, 1, 3); rv *= np.deg2rad(dr.cam_rot_deg) * s / (np.linalg.norm(rv) + 1e-9) * rng.uniform(0, 1)
        dq = np.zeros(4); mujoco.mju_axisAngle2Quat(dq, rv / (np.linalg.norm(rv) + 1e-12), np.linalg.norm(rv))
        q = np.zeros(4); mujoco.mju_mulQuat(q, m.cam_quat[cid], dq); m.cam_quat[cid] = q
        m.cam_fovy[cid] += rng.uniform(-1, 1) * dr.cam_fovy_deg * s
        m.light_diffuse[:] *= self._u(*dr.light_scale)
        m.light_diffuse[:] *= 1 + rng.uniform(-0.1, 0.1, m.light_diffuse.shape) * s
        # links (mass and inertia together)
        arm_bodies = [m.body(n).id for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")]
        f = 1 + rng.uniform(-1, 1, len(arm_bodies)) * dr.link_mass_frac * s
        m.body_mass[arm_bodies] *= f; m.body_inertia[arm_bodies] *= f[:, None]
        # cube
        cube = m.body("cube").id; g = m.geom("cube").id
        size = CUBE_HALF * (1 + rng.uniform(-1, 1) * dr.cube_size_frac * s)
        m.geom_size[g] = size
        mass = self._u(*dr.cube_mass)
        m.body_mass[cube] = mass; m.body_inertia[cube] = mass * (2 * size) ** 2 / 6
        m.geom_friction[g, 0] = self._u(*dr.cube_friction)
        # table tilt as a gravity tilt
        tilt = np.deg2rad(dr.table_tilt_deg * s) * rng.uniform(0, 1)
        az = rng.uniform(0, 2 * np.pi)
        m.opt.gravity[:] = 9.81 * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
        # textures and image pipeline
        if dr.textures and s > 0 and "wrist" in self.obs_keys:
            self._upload("wood", _wood(int(rng.integers(WOOD_POOL)), int(m.tex_width[m.texture("wood").id])))
            self._upload("rubik", textures.rubik_faces(int(rng.integers(1 << 30)), 96))
        self.img_fx = {
            "gain": self._u(*dr.image_gain), "gamma": self._u(*dr.image_gamma),
            "wb": 1 + rng.uniform(-1, 1, 3) * dr.image_wb * s, "blur": self._u(*dr.image_blur),
            "noise": self._u(*dr.image_noise), "jpeg": int(self._u(*dr.jpeg_quality)),
        }
        self._lut = None

    def _upload(self, name, rgb):
        m = self.sim.model
        tid = m.texture(name).id; adr = m.tex_adr[tid]; n = m.tex_height[tid] * m.tex_width[tid] * m.tex_nchannel[tid]
        rgb = np.asarray(rgb, np.uint8)
        if rgb.size != n:
            rgb = cv2.resize(rgb, (int(m.tex_width[tid]), int(m.tex_height[tid])))
        m.tex_data[adr:adr + n] = rgb.reshape(-1)
        self._tex_dirty = True

    # ------------------------------------------------------------------ sensing
    def _read_joints(self) -> np.ndarray:
        """What Present_Position would report: true pose seen through the calibration error, noise, ticks."""
        q = self.sim.read() - self.offset_err
        q = q + self._rng.normal(0, self.dr.encoder_noise_ticks * 360 / 4096 * self.dr.scale, 6)
        return self.contract.quantize(q)

    def _image(self) -> np.ndarray:
        w, h = self.image_size
        rw = int(min(640, max(160, 2 * max(w, h)))); rh = rw * 3 // 4      # the camera is 4:3 (640x480 DSHOW)
        rgb = self.sim.render(self.camera, rw, rh)
        if self._tex_dirty:
            r = self.sim._renderers[(rw, rh)]
            for name in ("wood", "rubik"):
                mujoco.mjr_uploadTexture(self.sim.model, r._mjr_context, self.sim.model.texture(name).id)
            self._tex_dirty = False
            rgb = self.sim.render(self.camera, rw, rh)
        fx = self.img_fx
        if self.dr.scale > 0:
            if self._lut is None:
                x = np.arange(256, dtype=np.float32) / 255.0
                wb = np.asarray(fx["wb"], np.float32) * np.float32(fx["gain"])
                self._lut = np.ascontiguousarray((np.clip(x[None, :] * wb[:, None], 0, 1) ** np.float32(fx["gamma"])).T.reshape(1, 256, 3))
            img = cv2.LUT(rgb, self._lut)
            if fx["blur"] > 0.05:
                img = cv2.GaussianBlur(img, (0, 0), fx["blur"])
            noise = self._rng.standard_normal(img.shape).astype(np.float32) * fx["noise"] * (0.8 * img + 0.6)
            img = np.clip(img + noise, 0, 1)
            out = (img * 255 + 0.5).astype(np.uint8)
            if fx["jpeg"] < 100:
                ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, fx["jpeg"]])
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
        else:
            out = rgb
        return preprocess_image(out, (w, h))

    def _obs(self) -> dict:
        o = {}
        if "joints" in self.obs_keys:
            o["joints"] = self._last_read.astype(np.float32)
        if "prev_goal" in self.obs_keys:
            o["prev_goal"] = self._goal.astype(np.float32)
        if "target" in self.obs_keys:
            o["target"] = np.zeros(3, np.float32) if self._target is None else self._target.astype(np.float32)
        if "wrist" in self.obs_keys:
            o["wrist"] = self._image()
        return o

    # ------------------------------------------------------------------ task
    def _sample_target(self) -> np.ndarray:
        """A reachable tip position: forward kinematics of a random in-range pose, kept 3-25 cm above the table."""
        m, d, c = self.sim.model, self.sim.data, self.contract
        saved = d.qpos.copy()
        for _ in range(200):
            pose = self._rng.uniform(c.low + 25, c.high - 25); pose[4] = c.start_pose[4]; pose[5] = 20
            d.qpos[self.sim.qadr] = c.to_model(pose); mujoco.mj_kinematics(m, d)
            tip = d.site("tip").xpos.copy()
            if 0.03 < tip[2] < 0.25 and 0.12 < tip[0] < 0.32 and abs(tip[1]) < 0.2:
                break
        d.qpos[:] = saved; mujoco.mj_forward(m, d)
        return tip

    def _place_cube(self):
        d, m = self.sim.data, self.sim.model
        adr = m.jnt_qposadr[m.joint("cube_free").id]
        r = self._rng.uniform(0.17, 0.28); a = self._rng.uniform(-0.6, 0.6)
        d.qpos[adr:adr + 3] = [r * np.cos(a), r * np.sin(a), m.geom_size[m.geom("cube").id][0] + 1e-4]
        yaw = self._rng.uniform(-np.pi, np.pi)
        d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[m.jnt_dofadr[m.joint("cube_free").id]:][:6] = 0
        badr = m.jnt_qposadr[m.joint("bowl_free").id]
        bowl = np.array([-4.0, -4.0])                                 # parked off the table, clear of other props
        if self._rng.uniform() < 0.5:                                 # half the time a distractor bowl
            for _ in range(50):
                rb, ab = self._rng.uniform(0.17, 0.32), self._rng.uniform(-0.8, 0.8)
                cand = np.array([rb * np.cos(ab), rb * np.sin(ab)])
                if np.linalg.norm(cand - d.qpos[adr:adr + 2]) > 0.15:
                    bowl = cand; break
        d.qpos[badr:badr + 3] = [bowl[0], bowl[1], 0.0]
        m.body_gravcomp[m.body("bowl").id] = 1.0 if abs(bowl[0]) > 1 else 0.0
        d.qpos[badr + 3:badr + 7] = [1, 0, 0, 0]
        d.qvel[m.jnt_dofadr[m.joint("bowl_free").id]:][:6] = 0

    def _cube_pos(self):
        return self.sim.data.body("cube").xpos.copy()

    def _reward(self) -> tuple[float, bool, dict]:
        tip = self.sim.site("tip")
        info = {}
        if self.task.name == "reach":
            dist = float(np.linalg.norm(tip - self._target_world))
            success = dist < self.task.success_radius
            r = -dist + (0.5 if success else 0.0)
            info["distance"] = dist
        elif self.task.name == "lift":
            cube = self._cube_pos()
            dist = float(np.linalg.norm(tip - cube))
            height = cube[2] - self._cube_rest_z
            success = height > self.task.lift_height
            r = -dist + 4.0 * np.clip(height, 0, self.task.lift_height * 1.5) / self.task.lift_height + (1.0 if success else 0.0)
            info.update(distance=dist, height=height)
        else:
            success, r = False, 0.0
        return float(r), bool(success), info

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = self.np_random
        options = options or {}
        self._randomise()
        start = np.asarray(options.get("start_pose", self.contract.start_pose), float)
        start = start + self._rng.uniform(-3, 3, 6) * self.dr.scale * np.r_[1, 0.3, 1, 1, 1, 0]
        start = self.contract.clip(start)
        # the simulator holds the TRUE model pose; the policy-facing frame is shifted by the calibration error
        self.sim.reset(start + self.offset_err)
        self._target, self._target_world = None, None
        if self.task.name == "reach":
            self._target_world = np.asarray(options.get("target", self._sample_target()), float)
            self._target = self._target_world.copy()
        if self.task.name == "reach":                       # nothing on the table to bump into
            for name in ("cube", "bowl"):
                adr = self.sim.model.jnt_qposadr[self.sim.model.joint(name + "_free").id]
                self.sim.data.qpos[adr:adr + 3] = [-3.0 - (name == "bowl"), -4.0, 0.0]   # clear of the parked cyl at (3,-4)
                self.sim.model.body_gravcomp[self.sim.model.body(name).id] = 1.0
        if self.task.name == "lift":
            self.sim.model.body_gravcomp[self.sim.model.body("cube").id] = 0.0
            self._place_cube()
            mujoco.mj_forward(self.sim.model, self.sim.data)
        self.sim.advance_to(0.3)            # let the arm settle under gravity with goal = start, cube on the table
        self._cube_rest_z = self._cube_pos()[2]
        self._t = self.sim.time
        self._goal = self.contract.clip(self.sim.goal() - self.offset_err)
        self._last_read = self._read_joints()
        self._steps = 0
        return self._obs(), {"privileged": self._privileged()}

    def _privileged(self) -> dict:
        return {"true_joints": self.sim.read(), "cube": self._cube_pos(), "tip": self.sim.site("tip"),
                "offset_err": self.offset_err.copy(), "latency": self.ep_params.latency}

    def action_to_goal(self, action: np.ndarray) -> np.ndarray:
        return goal_from_action(action, self.action_mode, self._goal, self._last_read, self.contract,
                                self.max_delta, self.max_relative_target)

    def step(self, action):
        goal = self.action_to_goal(action)
        s, dr = self.dr.scale, self.dr
        t_write = self._t + self._u(*dr.compute_delay)
        self.sim.advance_to(t_write)
        self.sim.command(goal + self.offset_err, t_write)
        self._goal = goal
        self._t += 1.0 / CONTROL_HZ + self._rng.uniform(-1, 1) * dr.period_jitter * s
        self.sim.advance_to(self._t)
        self._last_read = self._read_joints()
        self._steps += 1
        reward, success, info = self._reward()
        truncated = self._steps >= int(self.task.episode_seconds * CONTROL_HZ)
        info.update(success=success, privileged=self._privileged())
        return self._obs(), reward, False, truncated, info

    def render(self):
        return self.sim.render("overview", 640, 480)

    def close(self):
        self.sim.close()
