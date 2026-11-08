"""Record a system-identification run on the real arm.

  # dry run: builds the program, checks it in the sim, prints the plan. Touches nothing.
  python -m so101_sim2real.sysid.record --out runs/rec_A.npz --seed 1
  # the real thing, owner at the arm (about 4 minutes; create runs/STOP to abort):
  python -m so101_sim2real.sysid.record --out runs/rec_A.npz --seed 1 --port COM3 --i-am-at-the-arm
  # record a second run with another seed for the held-out check:
  python -m so101_sim2real.sysid.record --out runs/rec_B.npz --seed 2 --port COM3 --i-am-at-the-arm

Goals go out at 30 Hz on schedule; between them Present_Position is read as fast as the bus allows (every 4th
read also Present_Load). Every write and read is timestamped with perf_counter, which is what the replay uses.
Aborts (STOP file, Ctrl-C, tracking error > 35 deg for 1 s, |load| > 900 for 1.5 s) hold the measured pose
with torque on. The recording is saved even after an abort; completed segments remain usable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..contract import JOINTS, load_contract
from ..servo import prior
from ..sim import ArmSim
from .excite import Segment, build_program, check_program, find_home
from .log import Recording

TRACK_LIMIT_DEG = 35.0
LOAD_LIMIT = 900.0


def plan(seed: int, start_pose: np.ndarray | None = None, scale: float = 1.0):
    sim = ArmSim(prior())
    c = sim.contract
    home = find_home(sim)
    gravity = [home + np.array(d, float) for d in ([0, -25, 0, 0, 0, 0], [0, 15, -25, 0, 0, 0], [0, 0, 0, -30, 0, 0],
                                                    [0, -15, 20, 20, 0, 0])]
    segs = build_program(home, c.low, c.high, seed, gravity_poses=gravity, scale=scale)
    start = c.start_pose if start_pose is None else np.asarray(start_pose, float)
    n = int(np.ceil(np.max(np.abs(home - start)[:5]) / 15.0 * 30)) + 1
    approach = Segment("approach", -2, np.arange(n) / 30, np.array([start + (home - start) * k / (n - 1) for k in range(n)]))
    problems = check_program(sim, [approach] + segs)
    return home, segs, approach, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--id", default="follower_1")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink every amplitude (0.5 = half) for a first run")
    ap.add_argument("--joints", default="all", help="comma list of joint names to excite (default all)")
    ap.add_argument("--i-am-at-the-arm", action="store_true", help="required to move the arm")
    ap.add_argument("--rest", action="store_true", help="return to the folded start pose at the end")
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    stop = out.parent / "STOP"

    arm = None
    start_pose = None
    if a.i_am_at_the_arm:
        if stop.exists():
            raise SystemExit(f"{stop} exists: delete it first")
        from ..hw import Arm
        arm = Arm(a.port, a.id, allow_motion=True, stop_file=stop)
        start_pose = arm.read()
        print("present pose (owner deg):", np.round(start_pose, 1).tolist())
    home, segs, approach, problems = plan(a.seed, start_pose, a.scale)
    if a.joints != "all":
        keep = {JOINTS.index(j.strip()) for j in a.joints.split(",")}
        segs = [s for s in segs if s.joint in keep]
    total = sum(len(s.t) for s in segs) / 30
    print(f"home {np.round(home, 1).tolist()}; {len(segs)} segments, {total:.0f} s of excitation")
    for s in segs:
        print(f"  {s.name:>13}: {len(s.t) / 30:5.1f} s, range {np.round(s.goal.min(0) - home, 1).tolist()} .. {np.round(s.goal.max(0) - home, 1).tolist()}")
    if problems:
        print("SIM SAFETY CHECK FAILED, refusing to run:")
        for p in problems[:20]:
            print("  ", p)
        raise SystemExit(2)
    print("sim safety check: OK (gripper >= 30 mm above the table with 4 deg overshoot, >= 2 deg inside limits)")
    if arm is None:
        print("dry run only. Add --i-am-at-the-arm to execute.")
        return

    cmd_t, cmd, obs_t, obs, load, seg_log = [], [], [], [], [], []
    reason = "completed"
    track_bad = load_bad = 0.0
    T0 = time.perf_counter()

    def now():
        return time.perf_counter() - T0

    def read_once(k):
        t_a = now(); p = arm.read(); t_b = now()
        obs_t.append((t_a + t_b) / 2); obs.append(p)
        load.append(arm.read_load() if k % 4 == 0 else np.full(6, np.nan))
        return p

    try:
        print("approach to home at 15 deg/s ...")
        arm.move_to(approach.goal[-1])
        k = 0
        for seg in segs:
            t_seg = now() + 0.05
            seg_log.append({"name": seg.name, "joint": seg.joint, "t0": t_seg, "t1": None})
            last = None
            for i, (tc, g) in enumerate(zip(seg.t, seg.goal)):
                due = t_seg + tc
                while now() < due:              # read as fast as the bus allows until the next goal is due
                    last = read_once(k); k += 1
                if arm.stop_requested():
                    reason = "STOP file"; raise KeyboardInterrupt
                t_w = now(); arm.write(g); cmd_t.append(t_w); cmd.append(g.copy())
                if last is not None:
                    err = np.max(np.abs(last[:5] - g[:5]))
                    track_bad = track_bad + 1 / 30 if err > TRACK_LIMIT_DEG else 0.0
                    ld = load[-4:] if len(load) >= 4 else load
                    peak = np.nanmax(np.abs(np.array(ld))) if np.any(np.isfinite(ld)) else 0.0
                    load_bad = load_bad + 1 / 30 if peak > LOAD_LIMIT else 0.0
                    if track_bad > 1.0:
                        reason = f"tracking error {err:.0f} deg"; raise KeyboardInterrupt
                    if load_bad > 1.5:
                        reason = f"sustained load {peak:.0f}"; raise KeyboardInterrupt
            end = now() + 0.2
            while now() < end:
                read_once(k); k += 1
            seg_log[-1]["t1"] = now()
            print(f"  {seg.name} done")
    except KeyboardInterrupt:
        if reason == "completed":
            reason = "Ctrl-C"
        pos = arm.hold()
        print(f"ABORTED ({reason}); holding {np.round(pos, 1).tolist()} with torque on")
        if seg_log and seg_log[-1]["t1"] is None:
            seg_log.pop()                       # incomplete segment is not used for fitting
    finally:
        rec = Recording(np.array(cmd_t), np.array(cmd), np.array(obs_t), np.array(obs), np.array(load), seg_log,
                        {"source": "real", "port": a.port, "id": a.id, "seed": a.seed, "scale": a.scale,
                         "gains": arm.gains, "home": home.tolist(), "end": reason,
                         "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "read_hz": float(len(obs_t) / max(1e-6, (obs_t[-1] - obs_t[0]))) if len(obs_t) > 1 else 0.0})
        rec.save(out)
        print(f"saved {out}: {len(cmd_t)} goals, {len(obs_t)} reads ({rec.meta['read_hz']:.0f} Hz), {len(seg_log)} segments")
    if a.rest and reason == "completed":
        print("returning to the folded start pose; support the wrist")
        arm.move_to(start_pose)
    arm.close()
    print("port closed, torque left ON (the arm holds). Support it before disabling torque.")


if __name__ == "__main__":
    main()
