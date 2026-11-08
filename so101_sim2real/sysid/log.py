"""Recording format shared by the real recorder and the synthetic generator.

One .npz per recording:
  cmd_t  (N,)    seconds, perf_counter at the moment the Goal_Position sync_write was issued
  cmd    (N, 6)  owner-frame goals exactly as written (after clipping)
  obs_t  (M,)    seconds, midpoint of the Present_Position sync_read
  obs    (M, 6)  owner-frame readings
  load   (M, 6)  Present_Load (raw firmware units, sign = direction), NaN where not read
  segments       JSON list of {name, joint, t0, t1}
  meta           JSON dict (source, port, gains written, calibration, notes)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    cmd_t: np.ndarray
    cmd: np.ndarray
    obs_t: np.ndarray
    obs: np.ndarray
    load: np.ndarray
    segments: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, cmd_t=self.cmd_t, cmd=self.cmd, obs_t=self.obs_t, obs=self.obs, load=self.load,
                            segments=json.dumps(self.segments), meta=json.dumps(self.meta))

    @classmethod
    def load_npz(cls, path: Path) -> "Recording":
        z = np.load(path, allow_pickle=False)
        return cls(z["cmd_t"], z["cmd"], z["obs_t"], z["obs"], z["load"],
                   json.loads(str(z["segments"])), json.loads(str(z["meta"])))

    def segment(self, name: str) -> dict:
        return next(s for s in self.segments if s["name"] == name)
