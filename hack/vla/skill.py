"""VLA skill for the app (torch-free; runs in the repo .venv). The SmolVLA policy itself runs in a separate
process (hack/vla/server.py, LeRobot 0.6.1 venv, GPU) and is reached over a local socket.

    from hack.vla.skill import VLAClient, VLASkill, grab_views
    client = VLAClient()                                     # connects to the server (default 127.0.0.1:8765)
    skill = VLASkill(client, "put the red cube in the bowl")
    a = skill.act(obs, *grab_views(env))                     # (6,) in [-1, 1] for SO101Env(action_mode="delta", max_delta=6)

Pixels (front + wrist) + language + measured joints only: no object poses. The skill re-plans every `replan` ticks
and temporally ensembles the overlapping 50-step chunks (ACT-style), which keeps a history-free chunked policy
from stalling at the static phases of the scripted demos.
"""
from __future__ import annotations

import time
from multiprocessing.connection import Client

import numpy as np

MAX_DELTA = 6.0
ADDR = ("127.0.0.1", 8765)
AUTH = b"so101-vla"


def grab_views(env, size=256):
    """(wrist, front) uint8 RGB size x size exactly as in the demos: the env's wrist camera pipeline
    (contract.preprocess_image letterbox + image DR) and the fixed 'demo' camera."""
    old = env.image_size
    env.image_size = (size, size)
    try:
        wrist = env._image()
    finally:
        env.image_size = old
    return wrist, env.sim.render("demo", size, size)


def _pack(a):
    a = np.ascontiguousarray(a)
    return (a.tobytes(), a.shape, str(a.dtype))


class VLAClient:
    def __init__(self, addr=ADDR):
        self.conn = Client(addr, authkey=AUTH)

    def chunk(self, joints, wrist, front, instruction) -> np.ndarray:
        """(50, 6) absolute owner-frame joint goals."""
        self.conn.send({"joints": _pack(np.asarray(joints, np.float32)), "wrist": _pack(np.asarray(wrist, np.uint8)),
                        "front": _pack(np.asarray(front, np.uint8)), "instr": instruction})
        b, shape, dt = self.conn.recv()
        return np.frombuffer(b, dtype=dt).reshape(shape).astype(float)


class VLASkill:
    """One per instruction. act() once per 30 Hz tick."""

    def __init__(self, client, instruction: str, replan: int = 5, ens_k: float = 0.01, horizon: int = 50):
        self.c, self.instr, self.replan, self.k, self.h = client, instruction, replan, ens_k, horizon
        self.t = 0
        self.chunks: list[tuple[int, np.ndarray]] = []
        self.latency: list[float] = []

    def goal(self, joints, wrist, front) -> np.ndarray:
        if self.t % self.replan == 0 or not self.chunks:
            t0 = time.perf_counter()
            self.chunks.append((self.t, self.c.chunk(joints, wrist, front, self.instr)))
            self.latency.append(time.perf_counter() - t0)
        self.chunks = [(s, c) for s, c in self.chunks if self.t - s < min(len(c), self.h)]
        preds = np.stack([c[self.t - s] for s, c in self.chunks])          # oldest first
        w = np.exp(-self.k * np.arange(len(preds)))
        self.t += 1
        return (w[:, None] * preds).sum(0) / w.sum()

    def act(self, obs, wrist, front) -> np.ndarray:
        g = self.goal(obs["joints"], wrist, front)
        return np.clip((g - np.asarray(obs["prev_goal"], float)) / MAX_DELTA, -1, 1)
