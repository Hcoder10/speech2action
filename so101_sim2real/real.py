"""The real arm behind the same Gymnasium interface as `env.SO101Env`.

A policy trained in the sim runs here with no code change: same observation keys, same action mapping
(`contract.goal_from_action`), same image transform (`contract.preprocess_image`), same 30 Hz loop with the
observation read at the start of the tick and the goal written as soon as the policy returns.

  env = SO101RealEnv(task="reach", obs_keys=("joints", "prev_goal", "target"), allow_motion=True)
  obs, _ = env.reset(options={"target": [0.22, 0.05, 0.10]})    # metres, robot base frame (x forward, z up)
  while True: obs, r, term, trunc, info = env.step(policy(obs))

Reward is not computed on hardware (no ground truth); `info` carries timing so the sim's delay model can be
checked against what the loop actually did.
"""
from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .contract import CONTROL_HZ, goal_from_action, load_contract, preprocess_image
from .hw import Arm, grab_rgb, open_wrist_camera


class SO101RealEnv(gym.Env):
    def __init__(self, task: str = "reach", obs_keys=("joints", "prev_goal", "target"), action_mode: str = "delta",
                 max_delta: float = 6.0, max_relative_target: float | None = 10.0, image_size=(128, 128),
                 camera_index: int = 1, port: str = "COM3", robot_id: str = "follower_1", allow_motion: bool = False,
                 episode_seconds: float = 8.0, run_dir: str | Path = "runs"):
        super().__init__()
        self.contract = load_contract()
        self.obs_keys = tuple(obs_keys)
        self.action_mode, self.max_delta, self.max_relative_target = action_mode, max_delta, max_relative_target
        self.image_size = tuple(image_size)
        self.episode_steps = int(episode_seconds * CONTROL_HZ)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.arm = Arm(port, robot_id, allow_motion=allow_motion, stop_file=Path(run_dir) / "STOP")
        self.cap = open_wrist_camera(camera_index) if "wrist" in self.obs_keys else None
        lo, hi = (self.contract.low - 15).astype(np.float32), (self.contract.high + 15).astype(np.float32)   # calibration error + noise can read past the limits
        obs = {"joints": spaces.Box(lo, hi, dtype=np.float32), "prev_goal": spaces.Box(lo, hi, dtype=np.float32),
               "target": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
               "wrist": spaces.Box(0, 255, shape=(*self.image_size[::-1], 3), dtype=np.uint8)}
        self.observation_space = spaces.Dict({k: obs[k] for k in self.obs_keys})
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self._target = np.zeros(3)

    def _obs(self):
        o = {}
        if "joints" in self.obs_keys: o["joints"] = self._last_read.astype(np.float32)
        if "prev_goal" in self.obs_keys: o["prev_goal"] = self._goal.astype(np.float32)
        if "target" in self.obs_keys: o["target"] = self._target.astype(np.float32)
        if "wrist" in self.obs_keys: o["wrist"] = preprocess_image(grab_rgb(self.cap), self.image_size)
        return o

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if "start_pose" in options:
            self.arm.move_to(self.contract.clip(np.asarray(options["start_pose"], float)))
        if "target" in options:
            self._target = np.asarray(options["target"], float)
        self._last_read = self.arm.read()
        self._goal = self._last_read.copy()
        self._steps = 0
        self._tick = time.perf_counter()
        return self._obs(), {}

    def step(self, action):
        if self.arm.stop_requested():
            pos = self.arm.hold()
            raise RuntimeError(f"STOP file present; holding {np.round(pos, 1).tolist()}")
        goal = goal_from_action(action, self.action_mode, self._goal, self._last_read, self.contract,
                                self.max_delta, self.max_relative_target)
        t_write = time.perf_counter()
        self.arm.write(goal)
        self._goal = goal
        self._tick += 1.0 / CONTROL_HZ
        late = time.perf_counter() - self._tick
        while time.perf_counter() < self._tick:
            time.sleep(0.0005)
        if late > 0:                                   # overran the tick: resynchronise rather than burst
            self._tick = time.perf_counter()
        self._last_read = self.arm.read()
        obs = self._obs()
        self._steps += 1
        info = {"compute_delay": t_write - (self._tick - 1.0 / CONTROL_HZ), "overrun": max(0.0, late)}
        return obs, 0.0, False, self._steps >= self.episode_steps, info

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.arm.close()
