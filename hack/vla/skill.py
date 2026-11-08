"""VLA skill for the app (torch-free; runs in the repo .venv). The SmolVLA policy itself runs in a separate
process (hack/vla/server.py, LeRobot 0.6.1 venv, GPU) and is reached over a local socket.

    from hack.vla.skill import VLAClient, VLASkill, grab_views
    client = VLAClient()                                     # connects to the server (default 127.0.0.1:8777)
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
ADDR = ("127.0.0.1", 8777)
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
        self.async_ = False
        self._pending = None

    def start_async(self):
        """Real-time mode (the app's 30 Hz loop): inference runs in a background thread, the tick never blocks.
        A chunk is indexed by the tick of the observation it was computed from, so late chunks stay aligned."""
        import concurrent.futures as cf
        self.async_, self._pool = True, cf.ThreadPoolExecutor(1)
        return self

    def _request(self, joints, wrist, front, tick=None):
        t0, tick = time.perf_counter(), self.t if tick is None else tick
        c = self.c.chunk(joints, wrist, front, self.instr)
        self.latency.append(time.perf_counter() - t0)
        return tick, c

    def goal(self, joints, wrist, front=None) -> np.ndarray:
        """`wrist` may be a callable returning (wrist, front): views are then rendered only on ticks that re-plan."""
        if callable(wrist):
            views = wrist
            need = (not self.chunks or (self.async_ and self._pending is None and self.t - self.chunks[-1][0] >= self.replan)
                    or (not self.async_ and self.t % self.replan == 0))
            wrist, front = views() if need else (None, None)
        if self.async_:
            if self._pending is not None and self._pending.done():
                self.chunks.append(self._pending.result()); self._pending = None
            if not self.chunks:                       # first chunk: wait for it
                self.chunks.append(self._request(joints, wrist, front))
            elif self._pending is None and self.t - self.chunks[-1][0] >= self.replan:
                self._pending = self._pool.submit(self._request, np.array(joints), np.array(wrist), np.array(front), self.t)
        elif self.t % self.replan == 0 or not self.chunks:
            self.chunks.append(self._request(joints, wrist, front))
        live = [(s, c) for s, c in self.chunks if self.t - s < min(len(c), self.h)]
        self.chunks = live or self.chunks[-1:]
        preds = np.stack([c[min(self.t - s, len(c) - 1)] for s, c in self.chunks])          # oldest first
        w = np.exp(-self.k * np.arange(len(preds)))
        self.t += 1
        return (w[:, None] * preds).sum(0) / w.sum()

    def act(self, obs, wrist, front=None) -> np.ndarray:
        """obs from the env; (wrist, front) images, or a zero-arg callable returning them (rendered only when needed)."""
        g = self.goal(obs["joints"], wrist, front)
        return np.clip((g - np.asarray(obs["prev_goal"], float)) / MAX_DELTA, -1, 1)
