"""Low-level arm simulator: the calibrated v3 scene driven by the STS3215 model, on a real-time clock.

Time is explicit. `command(goal, t)` says "this goal was written to the bus at time t"; it reaches the servo
loop at t + latency. `advance_to(t)` integrates physics up to t. `read()` returns what Present_Position would
say right now. The real recorder logs exactly these three events, so a real log replays through this class
without any resampling (see `sysid.replay`).
"""
from __future__ import annotations

import bisect

import mujoco
import numpy as np

from .contract import ASSETS, JOINTS, Contract, load_contract
from .servo import ServoParams, prior


PARKED = ("cyl", "card", "block", "marker", "mug", "tape")
PROPS = PARKED + ("cube", "bowl")


class ArmSim:
    def __init__(self, params: ServoParams | None = None, contract: Contract | None = None,
                 scene: str | None = None, timestep: float = 0.002):
        self.contract = contract or load_contract()
        spec = mujoco.MjSpec.from_file(str(scene or ASSETS / "scene.xml"))
        for b in spec.bodies:      # MuJoCo only applies gravcomp at runtime to bodies that had it at compile time
            if b.name in PROPS:
                b.gravcomp = 1e-9
        self.model = spec.compile()
        self.model.opt.timestep = timestep
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.act = np.array([m.actuator(j).id for j in JOINTS])
        jnt = m.actuator_trnid[self.act, 0]
        self.qadr = m.jnt_qposadr[jnt]
        self.dadr = m.jnt_dofadr[jnt]
        self._renderers: dict = {}
        # props parked off the table (distractors, spare cylinder) would free-fall forever: float them in place
        for name in PARKED:
            m.body_gravcomp[m.body(name).id] = 1.0
        self._queue: list[tuple[float, int, np.ndarray]] = []   # (apply_time, seq, model-rad goal)
        self._seq = 0
        self._grip_stall = 0.0
        self._grip_overloaded = False
        self.set_params(params or prior())

    # ------------------------------------------------------------------ parameters
    def set_params(self, p: ServoParams) -> None:
        m = self.model
        self.params = p
        for i, (a, d) in enumerate(zip(self.act, self.dadr)):
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_biasprm[a, :] = 0.0
            m.actuator_gainprm[a, 0] = p.kp[i]
            m.actuator_biasprm[a, 1] = -p.kp[i]
            m.actuator_biasprm[a, 2] = -p.kv[i]
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-p.tau[i], p.tau[i])
            m.dof_damping[d] = p.tau[i] / p.omega_free[i] + p.damping[i]
            m.dof_frictionloss[d] = p.frictionloss[i]
            m.dof_armature[d] = p.armature[i]
        self._grip_stall = 0.0
        self._grip_overloaded = False

    # ------------------------------------------------------------------ state
    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self, owner_pose: np.ndarray, t: float = 0.0, owner_vel: np.ndarray | None = None) -> None:
        """Arm at `owner_pose` (owner degrees), goal = pose, empty command queue, objects untouched."""
        d = self.data
        q = self.contract.to_model(self.contract.clip(np.asarray(owner_pose, float)))
        d.qpos[self.qadr] = q
        d.qvel[self.dadr] = 0.0 if owner_vel is None else self.contract.vel_to_model(owner_vel)
        d.ctrl[self.act] = q
        d.time = t
        d.act[:] = 0.0
        self._queue.clear()
        self._grip_stall = 0.0
        self._grip_overloaded = False
        self.model.actuator_forcerange[self.act[5]] = (-self.params.tau[5], self.params.tau[5])
        mujoco.mj_forward(self.model, d)

    def command(self, owner_goal: np.ndarray, t: float | None = None) -> None:
        """Goal written to the bus at time t (default: now). Clipped to the owner's ranges like the real bus."""
        t = self.time if t is None else t
        goal = self.contract.to_model(self.contract.clip(np.asarray(owner_goal, float)))
        bisect.insort(self._queue, (t + self.params.latency, self._seq, goal))
        self._seq += 1

    def advance_to(self, t_end: float, chunk: float = 0.01) -> None:
        m, d = self.model, self.data
        dt = m.opt.timestep
        while d.time < t_end - 0.5 * dt:
            while self._queue and self._queue[0][0] <= d.time + 0.5 * dt:
                d.ctrl[self.act] = self._queue.pop(0)[2]
            horizon = t_end
            if self._queue:
                horizon = min(horizon, self._queue[0][0])
            horizon = min(horizon, d.time + chunk)
            n = max(1, int(round((horizon - d.time) / dt)))
            mujoco.mj_step(m, d, nstep=n)
            self._gripper_protection(n * dt)

    def _gripper_protection(self, elapsed: float) -> None:
        p, m, d = self.params, self.model, self.data
        a = self.act[5]
        demand = p.kp[5] * (d.ctrl[a] - d.qpos[self.qadr[5]]) - p.kv[5] * d.qvel[self.dadr[5]]
        if self._grip_overloaded:
            if abs(demand) < 0.9 * p.gripper_overload_frac * p.tau[5]:
                self._grip_overloaded = False
                self._grip_stall = 0.0
                m.actuator_forcerange[a] = (-p.tau[5], p.tau[5])
            return
        self._grip_stall = self._grip_stall + elapsed if abs(demand) >= 0.95 * p.tau[5] else 0.0
        if self._grip_stall >= p.gripper_overload_time:
            self._grip_overloaded = True
            lim = p.gripper_overload_frac * p.tau[5]
            m.actuator_forcerange[a] = (-lim, lim)

    # ------------------------------------------------------------------ readout
    def read(self) -> np.ndarray:
        """True joint positions in owner units (no noise, no quantisation)."""
        return self.contract.from_model(self.data.qpos[self.qadr])

    def read_vel(self) -> np.ndarray:
        v = self.data.qvel[self.dadr].copy()
        v[:5] = np.rad2deg(v[:5])
        v[5] = v[5] / self.contract.jaw_span_rad * 100.0
        return v

    def goal(self) -> np.ndarray:
        return self.contract.from_model(self.data.ctrl[self.act])

    def site(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def render(self, camera: str = "wrist", width: int = 640, height: int = 480) -> np.ndarray:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
