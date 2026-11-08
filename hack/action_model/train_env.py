"""Gymnasium env for the goal-conditioned reacher: SO101Env("reach") with full DR, plus moving targets, a `down`
flag, random start poses, and a reward on the TRUE tip. The observation is `kin.features(...)`, a function of
(joints, prev_goal, target, down) only, so it is computed identically on the real arm.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from so101_sim2real.contract import CONTROL_HZ
from so101_sim2real.env import DRConfig, SO101Env, TaskSpec

from .kin import FEAT_VERSION, OBS_DIM, Kin, chain_from_model, features

EPISODE_S = 6.0
SUCCESS_R = 0.015
DOWN_COS = float(np.cos(np.deg2rad(15)))
WS_LO = np.array([0.08, -0.26, 0.015])
WS_HI = np.array([0.36, 0.26, 0.26])
GRIP_GEOMS = ("fixed_pad_tip", "fixed_pad_mid", "palm", "moving_pad_tip", "moving_pad_mid")


def sample_cube_region(rng) -> np.ndarray:
    return np.array([rng.uniform(0.13, 0.32), rng.uniform(-0.20, 0.20), rng.uniform(0.02, 0.12)])


class ReachEnv(gym.Env):
    def __init__(self, dr: DRConfig | None = None, seed: int | None = None, p_down=0.5, p_redirect=0.5,
                 p_drift=0.35, p_random_start=0.5, chain: dict | None = None,
                 feat_version: int = FEAT_VERSION):
        super().__init__()
        self.env = SO101Env(TaskSpec("reach", episode_seconds=EPISODE_S), obs_keys=("joints", "prev_goal", "target"),
                            action_mode="delta", max_delta=6.0, max_relative_target=10.0, dr=dr, seed=seed)
        self.kin = Kin(chain or chain_from_model())
        self.c = self.env.contract
        self.feat_version = feat_version
        self.rng = np.random.default_rng(seed)
        self.p = dict(down=p_down, redirect=p_redirect, drift=p_drift, random_start=p_random_start)
        m = self.env.sim.model
        self._grip = {m.geom(n).id for n in GRIP_GEOMS}
        self._table = m.geom("table").id
        self._gripper_body = m.body("gripper").id
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM[feat_version],), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.forced = None   # eval hook: dict(down=, redirect=, drift=, random_start=) overrides the coin flips

    # ---------------------------------------------------------------- sampling
    def _sample_target(self, down: bool) -> np.ndarray:
        if down or self.rng.uniform() < 0.5:
            return sample_cube_region(self.rng)
        return self.env._sample_target()

    def _start_pose(self, random: bool) -> np.ndarray:
        c = self.c
        pose = c.start_pose.copy()
        if random:
            for _ in range(100):
                p = pose.copy()
                p[:4] = self.rng.uniform(c.low[:4] + 15, c.high[:4] - 15)
                tip, R = self.kin.fk(p[:5])
                wrist = tip + R @ np.array([0, 0, 0.104])       # gripper origin
                if tip[2] > 0.03 and wrist[2] > 0.06 and tip[0] > 0.0 and np.hypot(tip[0], tip[1]) > 0.10:
                    pose = p
                    break
        if self.rng.uniform() < 0.5:
            pose[4] = self.rng.uniform(-150, 150)                 # roll is the skill layer's; see all of it
        pose[5] = self.rng.uniform(0, 100)
        return pose

    def _flip(self, key):
        if self.forced is not None and key in self.forced:
            return bool(self.forced[key])
        return bool(self.rng.uniform() < self.p[key])

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.down = self._flip("down")
        self.base_target = self._sample_target(self.down)
        n = int(EPISODE_S * CONTROL_HZ)
        self.redirect_at = int(self.rng.integers(30, 120)) if self._flip("redirect") else -1
        self.drift = self._flip("drift")
        self.drift_from = self.drift_to = np.zeros(3); self.drift_t0 = 0; self.drift_len = 1
        start = self._start_pose(self._flip("random_start"))
        obs, info = self.env.reset(seed=None if seed is None else int(seed), options={"start_pose": start, "target": self.base_target})
        self.t = 0; self.n = n
        self.target = self.base_target.copy()
        self.prev_a = np.zeros(4)
        self.joints, self.goal = obs["joints"], obs["prev_goal"]
        return self._obs(), {"privileged": info["privileged"]}

    def _obs(self):
        return features(self.kin, self.joints, self.goal, self.target, float(self.down), self.feat_version)

    def _advance_target(self):
        if self.t == self.redirect_at:
            self.base_target = self._sample_target(self.down)
            self.drift_from = self.drift_to = np.zeros(3)
        if self.drift:
            if self.t >= self.drift_t0 + self.drift_len:
                self.drift_from = self.drift_to
                v = self.rng.normal(size=3); v /= np.linalg.norm(v)
                self.drift_to = self.drift_from + v * self.rng.uniform(0.01, 0.04)
                self.drift_t0 = self.t; self.drift_len = int(self.rng.uniform(0.5, 1.5) * CONTROL_HZ)
            a = (self.t - self.drift_t0) / self.drift_len
            off = self.drift_from + a * (self.drift_to - self.drift_from)
        else:
            off = 0.0
        self.target = np.clip(self.base_target + off, WS_LO, WS_HI)
        if self.down:
            self.target[2] = max(self.target[2], 0.02)

    def _contact(self) -> bool:
        d = self.env.sim.data
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 == self._table and g2 in self._grip) or (g2 == self._table and g1 in self._grip):
                return True
        return False

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        full = np.zeros(6); full[:4] = a
        target_now = self.target.copy()                  # what the policy was asked to reach this tick
        obs, _, _, trunc, info = self.env.step(full)
        self.joints, self.goal = obs["joints"], obs["prev_goal"]
        tip = info["privileged"]["tip"]
        R = self.env.sim.data.xmat[self._gripper_body].reshape(3, 3)
        align = float(R[2, 2])                           # finger dir is -Z_gripper; dot with world -Z
        dist = float(np.linalg.norm(tip - target_now))
        contact = self._contact()
        ok = dist < SUCCESS_R and (not self.down or align > DOWN_COS)
        r = -5.0 * dist + 0.5 * (1.0 - np.tanh(dist / 0.01)) + 0.5 * (1.0 - np.tanh(dist / 0.003))
        if self.down:
            r -= 1.0 * float(np.arccos(np.clip(align, -1, 1))) * (0.5 + 0.5 * np.exp(-dist / 0.03))
        r -= 0.03 * float(np.sum((a - self.prev_a) ** 2)) + 0.005 * float(np.sum(a ** 2))
        r -= 2.0 * contact
        r += 1.0 * ok
        self.prev_a = a
        self.t += 1
        self._advance_target()
        info = {"dist": dist, "align": align, "success": ok, "contact": contact, "down": self.down,
                "redirect_at": self.redirect_at, "t": self.t, "privileged": info["privileged"]}
        return self._obs(), float(r), False, bool(trunc), info


def make_env(rank: int, seed: int = 0, dr_scale: float = 1.0):
    def _f():
        return ReachEnv(dr=DRConfig(scale=dr_scale) if dr_scale != 1.0 else DRConfig(), seed=seed + rank)
    return _f
