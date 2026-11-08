"""Learned goal-conditioned reacher, numpy-only inference (no torch, no mujoco at runtime).

    m = ActionModel()                       # loads hack/action_model/reacher.npz
    a = m.act(obs["joints"], obs["prev_goal"], target_xyz, down=1.0)
    env.step(a)                             # SO101Env(action_mode="delta", max_delta=6.0); set a[4], a[5] yourself

The observation is built from (joints, prev_goal, target, down) with nominal forward kinematics (`kin.py`), the
same function the training env uses, so sim and real arm feed it identically.

Notes for the skill layer (v2, PPO 5.6M steps, see eval.py):
  * Converges to 15 mm in ~0.6 s (free) / ~0.9 s (down) and re-acquires a jumped target in ~0.3 s; call it every
    tick with the CURRENT target, moving targets are fine (trained on 1-4 cm/s drifts and jumps).
  * Precision: ~3 mm without calibration error; under full DR ~13 mm median, and that is entirely the per-episode
    joint-zero error (DR minus that term: 3-4 mm). It is a fixed bias per arm, so a constant target offset
    measured once on the real arm removes most of it.
  * down=1 is kinematically achievable (fingers vertical) only for tip heights <= ~6 cm within ~0.3 m reach;
    at 9-12 cm the arm cannot point straight down (min tilt 5-30 deg), so hover low (~5 cm above the grasp).
  * Roll (a[4]) and gripper (a[5]) are 0 here; the model sees the measured roll and adapts the tip to it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .kin import Kin, features

DEFAULT = Path(__file__).resolve().parent / "reacher.npz"


class ActionModel:
    def __init__(self, path: str | Path = DEFAULT):
        z = np.load(path, allow_pickle=False)
        n = int(z["n_hidden"])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.Wa, self.ba = z["Wa"], z["ba"]
        self.mean, self.var = z["obs_mean"], z["obs_var"]
        self.eps, self.clip = float(z["obs_eps"]), float(z["obs_clip"])
        self.kin = Kin({k: z[k] for k in ("kin_pos", "kin_rot", "kin_offset_deg", "kin_tip")})
        self.meta = str(z["meta"]) if "meta" in z else ""
        self.feat_version = int(z["feat_version"]) if "feat_version" in z else 1

    def reset(self) -> None:
        pass                                   # stateless policy

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)

    def forward(self, nobs: np.ndarray) -> np.ndarray:
        """Normalised observation (..., D) -> deterministic mean action (..., 4), clipped to [-1, 1]."""
        h = np.asarray(nobs, np.float64)
        for W, b in self.layers:
            h = np.tanh(h @ W.T + b)
        return np.clip(h @ self.Wa.T + self.ba, -1.0, 1.0)

    def act(self, joints: np.ndarray, prev_goal: np.ndarray, target: np.ndarray, down: float) -> np.ndarray:
        obs = features(self.kin, joints, prev_goal, target, down, self.feat_version).astype(np.float64)
        out = np.zeros(6)
        out[:4] = self.forward(self.normalize(obs))
        return out
