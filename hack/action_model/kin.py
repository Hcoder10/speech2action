"""Numpy-only forward kinematics of the SO-101 on the NOMINAL model (owner-frame degrees in, base-frame metres out).

Every policy feature built from it is a function of the measured joints alone, so it is exactly as available on
the real arm as in the sim. (It inherits the calibration error the sim randomises, which nothing can observe.)
"""
from __future__ import annotations

import numpy as np

BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper")   # each carries one z-axis hinge at its origin


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def chain_from_model() -> dict:
    """Pull the fixed chain out of the compiled MuJoCo scene (needs mujoco; run at training/export time only)."""
    from so101_sim2real.contract import load_contract
    from so101_sim2real.sim import ArmSim
    s = ArmSim(); m = s.model; c = load_contract()
    for n in BODIES:
        j = m.body(n).jntadr[0]
        assert np.allclose(m.jnt_pos[j], 0) and np.allclose(m.jnt_axis[j], [0, 0, 1]) and m.qpos0[m.jnt_qposadr[j]] == 0
    base = m.body("base")
    assert np.allclose(base.pos, 0) and np.allclose(base.quat, [1, 0, 0, 0])
    return {
        "kin_pos": np.array([m.body_pos[m.body(n).id] for n in BODIES]),
        "kin_rot": np.array([_quat_mat(m.body_quat[m.body(n).id]) for n in BODIES]),
        "kin_offset_deg": np.asarray(c.offset_deg[:5], float),
        "kin_tip": np.asarray(m.site_pos[m.site("tip").id], float),
    }


class Kin:
    def __init__(self, chain: dict):
        self.pos = np.asarray(chain["kin_pos"], float)
        self.rot = np.asarray(chain["kin_rot"], float)
        self.off = np.asarray(chain["kin_offset_deg"], float)
        self.tip_local = np.asarray(chain["kin_tip"], float)

    def fk(self, owner_deg) -> tuple[np.ndarray, np.ndarray]:
        """(tip position (3,), gripper rotation matrix (3,3)) for owner-frame joints[0:5] in degrees."""
        q = np.deg2rad(np.asarray(owner_deg, float)[:5] + self.off)
        R = np.eye(3); p = np.zeros(3)
        for i in range(5):
            p = p + R @ self.pos[i]
            c, s = np.cos(q[i]), np.sin(q[i])
            R = R @ self.rot[i] @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return p + R @ self.tip_local, R


OBS_DIM = {1: 30, 2: 36}
FEAT_VERSION = 2


def features(kin: Kin, joints, prev_goal, target, down, version: int = FEAT_VERSION) -> np.ndarray:
    """The policy observation. Inputs are exactly what the real arm + skill layer provide (owner degrees, metres).
    v2 adds saturating high-gain copies of the two tip errors (cm-scale precision for the MLP)."""
    j = np.asarray(joints, float); g = np.asarray(prev_goal, float); t = np.asarray(target, float)
    tip, R = kin.fk(j[:5])
    tip_g, _ = kin.fk(np.r_[g[:4], j[4]])
    parts = [
        j[:6] / 100.0,                 # pan, lift, elbow, wrist_flex, roll, gripper
        g[:4] / 100.0,
        (g[:4] - j[:4]) / 10.0,        # commanded-vs-measured lag (servo state proxy)
        t,
        tip,                           # nominal FK of the measured joints
        (t - tip) * 10.0,
        (t - tip_g) * 10.0,            # error of where the last goal is heading
        -R[:, 2],                      # finger direction (nominal)
        [float(down)],
    ]
    if version >= 2:
        parts += [np.tanh((t - tip) / 0.01), np.tanh((t - tip_g) / 0.01)]
    return np.concatenate(parts).astype(np.float32)
