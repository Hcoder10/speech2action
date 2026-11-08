"""Replay a recording's commands through the simulator and predict what Present_Position would have read.

Also generates synthetic recordings from a known ServoParams, which is how the fit is tested without hardware.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..contract import CONTROL_HZ
from ..servo import ServoParams
from ..sim import ArmSim
from .excite import Segment
from .log import Recording


def make_sim(params: ServoParams) -> ArmSim:
    """Arm-only dynamics: contacts off (sysid never touches anything), so a replay runs in milliseconds."""
    sim = ArmSim(params)
    sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return sim


def replay_segment(sim: ArmSim, rec: Recording, seg: dict, max_obs_hz: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (obs_index, predicted owner-frame positions) for the observations inside the segment."""
    t0, t1 = seg["t0"], seg["t1"]
    oi = np.flatnonzero((rec.obs_t >= t0) & (rec.obs_t <= t1))
    if max_obs_hz:
        keep, last = [], -np.inf
        for i in oi:
            if rec.obs_t[i] - last >= 1.0 / max_obs_hz - 1e-9:
                keep.append(i); last = rec.obs_t[i]
        oi = np.array(keep, int)
    first = oi[0]
    sim.reset(rec.obs[first], t=rec.obs_t[first])
    prior_cmd = np.flatnonzero(rec.cmd_t <= rec.obs_t[first])
    if prior_cmd.size:
        sim.data.ctrl[sim.act] = sim.contract.to_model(sim.contract.clip(rec.cmd[prior_cmd[-1]]))
    ci = np.flatnonzero((rec.cmd_t > rec.obs_t[first]) & (rec.cmd_t <= t1))
    events = sorted([(rec.cmd_t[i], 0, i) for i in ci] + [(rec.obs_t[i], 1, i) for i in oi[1:]])
    pred = np.empty((oi.size, 6))
    pred[0] = rec.obs[first]
    k = 1
    for t, kind, i in events:
        sim.advance_to(t)
        if kind == 0:
            sim.command(rec.cmd[i], t)
        else:
            pred[k] = sim.read(); k += 1
    return oi, pred


def synthesize(params: ServoParams, home: np.ndarray, segments: list[Segment], seed: int = 0,
               obs_hz: float = 120.0, jitter: float = 0.002, quantize: bool = True, noise_ticks: float = 0.5,
               pre_settle: float = 1.0) -> Recording:
    """What the real recorder would log if the arm were exactly `params`: 30 Hz goals, ~obs_hz reads with timing
    jitter, tick quantisation and encoder noise. Segments are laid end to end with the arm settling at home."""
    rng = np.random.default_rng(seed)
    sim = make_sim(params)
    sim.reset(home, t=0.0)
    sim.advance_to(pre_settle)
    cmd_t, cmd, obs_t, obs, segs = [], [], [], [], []
    t = pre_settle
    next_obs = t
    for seg in segments:
        t0 = t
        times = t0 + seg.t
        for k, (tc, g) in enumerate(zip(times, seg.goal)):
            tc_j = tc + rng.uniform(0, jitter)
            while next_obs < tc_j:
                sim.advance_to(next_obs)
                obs_t.append(next_obs); obs.append(sim.read())
                next_obs += 1.0 / obs_hz + rng.uniform(-jitter, jitter) / 4
            sim.advance_to(tc_j)
            sim.command(g, tc_j)
            cmd_t.append(tc_j); cmd.append(sim.contract.clip(g))
        t = times[-1] + 1.0 / CONTROL_HZ
        segs.append({"name": seg.name, "joint": seg.joint, "t0": float(t0), "t1": float(t)})
    while next_obs < t:
        sim.advance_to(next_obs); obs_t.append(next_obs); obs.append(sim.read()); next_obs += 1.0 / obs_hz
    obs = np.array(obs)
    if noise_ticks:
        obs = obs + rng.normal(0, noise_ticks * 360 / 4096, obs.shape)
    if quantize:
        obs = sim.contract.quantize(obs)
    return Recording(np.array(cmd_t), np.array(cmd), np.array(obs_t), obs, np.full_like(obs, np.nan), segs,
                     {"source": "synthetic", "truth": params.to_json(), "seed": seed})
