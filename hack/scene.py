"""Demo world: the calibrated SO-101 sim with three coloured cubes and the white bowl in front of the arm.

`DemoEnv` is `SO101Env` (same servo model, latency, sensing) on a scene with the extra cubes. Object poses are
read from the sim; on the real arm the same names come from a fixed layout instead.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from so101_sim2real.contract import ASSETS
from so101_sim2real.env import DRConfig, SO101Env
from so101_sim2real.sim import ArmSim

HERE = Path(__file__).resolve().parent
COLORS = {"red": (0.85, 0.12, 0.12, 1), "blue": (0.12, 0.35, 0.90, 1), "green": (0.10, 0.70, 0.25, 1)}
CUBE_HALF = float(__import__("os").environ.get("HACK_CUBE_HALF", 0.015))
BOWL_RIM_R = 0.075


def build_scene() -> Path:
    """scene.xml + three cubes, written next to the vendored meshes so relative paths keep working."""
    out = ASSETS / "demo_scene.xml"
    spec = mujoco.MjSpec.from_file(str(ASSETS / "scene.xml"))
    for i, (name, rgba) in enumerate(COLORS.items()):
        b = spec.worldbody.add_body(name=f"{name}_cube", pos=[-2.0, -4.0 + 0.3 * i, CUBE_HALF])   # parked; reset places it
        b.add_freejoint(name=f"{name}_free")
        b.add_geom(name=f"{name}_cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, rgba=list(rgba),
                   mass=0.03, friction=[1.2, 0.02, 0.001], condim=6, solimp=[0.95, 0.99, 0.001, 0.5, 2],
                   solref=[0.004, 1])
        b.gravcomp = 1e-9
    cam = spec.worldbody.add_camera(name="demo", pos=[0.70, 0.0, 0.50], fovy=48)   # facing the arm: screen left = "left"
    cam.targetbody = "shoulder"
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    spec.compile()
    xml = spec.to_xml()
    if not out.exists() or out.read_text() != xml:           # atomic: parallel processes may build at once
        import os, tempfile
        fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(xml)
        os.replace(tmp, out)
    return out


class DemoEnv(SO101Env):
    def __init__(self, dr: DRConfig | None = None, seed: int | None = None, **kw):
        kw.setdefault("obs_keys", ("joints", "prev_goal"))
        super().__init__(task="free", dr=dr if dr is not None else DRConfig(scale=0.3, textures=False), seed=seed, **kw)
        scene = build_scene()
        self.sim.close()
        self.sim = ArmSim(self.nominal_params.copy(), self.contract, scene=str(scene))
        for name in COLORS:                      # float the parked cubes until reset() puts them on the table
            self.sim.model.body_gravcomp[self.sim.model.body(f"{name}_cube").id] = 1.0
        self._nominal = self._snapshot_model()
        self.task.episode_seconds = 1e9

    # ------------------------------------------------------------------ layout
    def _set_free(self, body: str, xyz, yaw: float = 0.0, park: bool = False):
        m, d = self.sim.model, self.sim.data
        j = m.body(body).jntadr[0]
        a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
        d.qpos[a:a + 3] = xyz
        d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[v:v + 6] = 0
        m.body_gravcomp[m.body(body).id] = 1.0 if park else 0.0

    def layout(self, rng: np.random.Generator) -> dict:
        """Bowl on one side, cubes spread across the band where top-down grasps are reliable, nothing overlapping."""
        for _attempt in range(100):
            placed = {}
            bowl_y = rng.choice([-1, 1]) * rng.uniform(0.14, 0.18)
            placed["bowl"] = np.array([rng.uniform(0.20, 0.24), bowl_y])
            for name in COLORS:
                for _ in range(100):
                    r, a = rng.uniform(0.19, 0.28), rng.uniform(-0.75, 0.75)
                    p = np.array([r * np.cos(a), r * np.sin(a)])
                    if all(np.linalg.norm(p - q) > (0.12 if k == "bowl" else 0.07) for k, q in placed.items()):
                        placed[name] = p
                        break
            if len(placed) == 1 + len(COLORS):
                return placed
        raise RuntimeError("could not lay out the scene")

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._set_free("cube", [-3.0, -4.0, 0.0], park=True)
        lay = (options or {}).get("layout") or self.layout(self._rng)
        self._set_free("bowl", [*lay["bowl"], 0.0])
        for name in COLORS:
            self._set_free(f"{name}_cube", [*lay[name], CUBE_HALF + 1e-4], yaw=self._rng.uniform(-0.6, 0.6))
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.sim.advance_to(self.sim.time + 0.3)
        self._t = self.sim.time
        self._last_read = self._read_joints()
        return self._obs(), {"privileged": self._privileged()}

    # ------------------------------------------------------------------ world state for the skill layer
    def objects(self) -> dict:
        d, m = self.sim.data, self.sim.model
        out = {}
        for name in COLORS:
            b = m.body(f"{name}_cube").id
            R = d.xmat[b].reshape(3, 3)
            out[name] = {"pos": d.xpos[b].copy(), "yaw": float(np.arctan2(R[1, 0], R[0, 0])), "size": 2 * CUBE_HALF}
        out["bowl"] = {"pos": d.body("bowl").xpos.copy(), "yaw": 0.0, "size": 2 * BOWL_RIM_R}
        return out

    def render_view(self, camera="demo", w=960, h=720):
        return self.sim.render(camera, w, h)
