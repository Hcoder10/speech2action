"""Excitation programs for system identification, and the sim-side safety check every program must pass.

A program is a list of segments; each segment moves ONE joint about a hold pose while the others hold, so each
joint's parameters are identifiable from its own segment. Content per joint, designed for the three regimes the
servo model has:
  small steps (2, 5 deg)       -> linear regime: kp, kv, armature, friction (stiction shows as a dead band)
  large steps (12, 25 deg)     -> torque/speed saturation: tau, omega_free
  log chirp 0.3 -> 4 Hz        -> latency and phase (bandwidth)
  constant-speed ramps         -> viscous vs Coulomb friction
  gravity holds (pitch joints) -> stiffness under load (steady-state sag = gravity torque / kp)
Commands are emitted at the deployment rate (30 Hz), never faster, so the fit sees the policy's own input rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import CONTROL_HZ, JOINTS


@dataclass
class Segment:
    name: str
    joint: int
    t: np.ndarray        # (N,) command times relative to segment start, seconds
    goal: np.ndarray     # (N, 6) owner-frame goals


def _hold(pose, secs):
    n = max(1, int(round(secs * CONTROL_HZ)))
    return [pose.copy() for _ in range(n)]


def joint_program(home: np.ndarray, j: int, low: np.ndarray, high: np.ndarray, rng: np.random.Generator,
                  margin: float = 8.0, scale: float = 1.0) -> Segment:
    home = np.asarray(home, float)
    room_up = high[j] - margin - home[j]
    room_dn = home[j] - (low[j] + margin)
    room = max(0.0, min(room_up, room_dn))
    frames: list[np.ndarray] = _hold(home, 1.0)

    def at(delta):
        p = home.copy(); p[j] = home[j] + delta; return p

    if j == 5:  # gripper, 0..100 units: open/close steps from the current opening
        levels = rng.permutation([35.0, 60.0, 90.0])
        for lv in levels:
            frames += _hold(at(np.clip(lv - home[j], -home[j], 100 - home[j])), 1.0)
            frames += _hold(home, 1.0)
        return Segment(JOINTS[j], j, np.arange(len(frames)) / CONTROL_HZ, np.array(frames))

    amps = [a * scale for a in (2.0, 5.0, 12.0, 25.0) if a * scale <= room]
    for a in rng.permutation(amps):
        for sign in rng.permutation([1.0, -1.0]):
            frames += _hold(at(sign * a), 0.8)
            frames += _hold(home, 0.8)
    amp = min(8.0 * scale, room)
    if amp > 1.0:
        T = 8.0
        t = np.arange(int(T * CONTROL_HZ)) / CONTROL_HZ
        f0, f1 = 0.3, 4.0
        phase = 2 * np.pi * f0 * T / np.log(f1 / f0) * (np.exp(t / T * np.log(f1 / f0)) - 1) + rng.uniform(0, 2 * np.pi)
        env = np.minimum(1.0, np.minimum(t, T - t) / 0.5)
        frames += [at(amp * e * np.sin(ph)) for e, ph in zip(env, phase)]
        frames += _hold(home, 0.8)
    ramp = min(20.0 * scale, room)
    if ramp > 2.0:
        for speed in rng.permutation([30.0, 90.0, 180.0]):
            n = max(2, int(round(ramp / speed * CONTROL_HZ)))
            for target in (ramp, -ramp, 0.0):
                start = frames[-1][j] - home[j]
                frames += [at(start + (target - start) * (k + 1) / n) for k in range(n)]
                frames += _hold(at(target), 0.4)
    frames += _hold(home, 0.8)
    return Segment(JOINTS[j], j, np.arange(len(frames)) / CONTROL_HZ, np.array(frames))


def gravity_program(home: np.ndarray, poses: list[np.ndarray], speed: float = 20.0, hold: float = 1.5) -> Segment:
    """Slow moves between poses, each held: steady-state sag of lift/elbow/wrist against gravity."""
    frames = _hold(np.asarray(home, float), 0.5)
    for p in list(poses) + [np.asarray(home, float)]:
        a = frames[-1]
        n = max(2, int(np.ceil(np.max(np.abs(p - a)) / speed * CONTROL_HZ)))
        frames += [a + (p - a) * (k + 1) / n for k in range(n)]
        frames += _hold(p, hold)
    return Segment("gravity", -1, np.arange(len(frames)) / CONTROL_HZ, np.array(frames))


def build_program(home: np.ndarray, low: np.ndarray, high: np.ndarray, seed: int, gravity_poses=None,
                  joints=range(6), scale: float = 1.0) -> list[Segment]:
    rng = np.random.default_rng(seed)
    segs = [joint_program(home, j, low, high, rng, scale=scale) for j in joints]
    if gravity_poses:
        segs.append(gravity_program(home, gravity_poses))
    return segs


# ---------------------------------------------------------------------------------------------- safety
CLEARANCE_GEOMS = ("fixed_pad_tip", "moving_pad_tip", "fixed_pad_mid", "moving_pad_mid", "palm")


def check_program(sim, segments: list[Segment], min_table_mm: float = 30.0, overshoot_deg: float = 4.0) -> list[str]:
    """Kinematic check of every goal (plus an overshoot allowance on the moving joint) in the sim.

    Returns a list of problems; empty means the program may be sent. Uses the bench-fitted table plane (z=0)
    and the gripper collision primitives; the real recorder refuses to start unless this is empty.
    """
    import mujoco

    m, d = sim.model, sim.data
    ids = [m.geom(g).id for g in CLEARANCE_GEOMS]
    problems = []
    c = sim.contract
    for seg in segments:
        for k, goal in enumerate(seg.goal):
            for extra in (0.0, overshoot_deg, -overshoot_deg):
                g = goal.copy()
                if seg.joint >= 0 and seg.joint < 5:
                    g[seg.joint] += extra
                if seg.joint != -2 and (np.any(g[:5] < c.low[:5] + 2) or np.any(g[:5] > c.high[:5] - 2)):
                    # (joint -2 = the approach from wherever the arm physically rests, often against its stops)
                    problems.append(f"{seg.name}[{k}] within 2 deg of a joint limit: {np.round(g, 1).tolist()}")
                    break
                d.qpos[sim.qadr] = c.to_model(g)
                mujoco.mj_kinematics(m, d)
                zmin = min(d.geom_xpos[i][2] - np.max(m.geom_size[i]) for i in ids)
                if zmin * 1000 < min_table_mm:
                    problems.append(f"{seg.name}[{k}] gripper {zmin * 1000:.0f} mm above table (< {min_table_mm})")
                    break
            if len(problems) > 20:
                return problems
    return problems


def find_home(sim, rng_seed: int = 0) -> np.ndarray:
    """A hold pose with the most room for excitation: tip 14-22 cm up and 16-26 cm out, roll level, jaw 20.

    Deterministic grid search over lift/elbow/wrist_flex in the owner frame using the calibrated model.
    """
    import mujoco

    c, m, d = sim.contract, sim.model, sim.data
    level_roll = float(c.start_pose[4])
    best, best_score = None, -np.inf
    for lift in np.arange(-80, 81, 5.0):
        for elbow in np.arange(-90, 91, 5.0):
            for wrist in np.arange(-90, 91, 5.0):
                pose = np.array([0.0, lift, elbow, wrist, level_roll, 20.0])
                d.qpos[sim.qadr] = c.to_model(pose)
                mujoco.mj_kinematics(m, d)
                tip = d.site("tip").xpos
                if not (0.14 <= tip[2] <= 0.22 and 0.16 <= tip[0] <= 0.26 and abs(tip[1]) < 0.03):
                    continue
                room = np.minimum(pose[1:4] - c.low[1:4], c.high[1:4] - pose[1:4])
                score = room.min()
                if score > best_score:
                    best, best_score = pose, score
    if best is None:
        raise RuntimeError("no hold pose found in the search box")
    return best
