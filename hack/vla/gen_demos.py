"""Expert demos for the VLA skill: privileged-pose scripted expert (IK skills) -> pixels + joints + language episodes.

Per 30 Hz tick: wrist image (env DR'd camera, contract.preprocess_image letterbox, 256x256), fixed front view
("demo" camera, 256x256), state = obs["joints"] (owner degrees, gripper 0..100), prev_goal before the step,
action = absolute joint goal the env applied (obs["prev_goal"] after the step), the task string, expert phase.
Only successful episodes (named cube ends in the bowl) are written.

  python -m hack.vla.gen_demos --out runs/vla_demos --seeds 10000:10700 --workers 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

COLORS = ("red", "blue", "green")
TEMPLATES = ("pick up the {c} cube and put it in the bowl", "put the {c} cube in the bowl")


def task_for(seed: int):
    rng = np.random.default_rng(seed + 7)
    c = COLORS[int(rng.integers(3))]
    return c, TEMPLATES[int(rng.integers(len(TEMPLATES)))].format(c=c)


def in_bowl(objs, name):
    return bool(np.linalg.norm(objs[name]["pos"][:2] - objs["bowl"]["pos"][:2]) < 0.06 and objs[name]["pos"][2] < 0.06)


def _jpg(img):
    import cv2
    ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return enc.reshape(-1)


def run_episode(seed: int, out_dir: str, max_s: float = 25.0):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from hack.intent import Intent
    from hack.scene import DemoEnv
    from hack.vla.expert_snapshot import IKController, Kinematics, Skills

    color, task = task_for(seed)
    env = DemoEnv(seed=seed, obs_keys=("joints", "prev_goal", "wrist"), image_size=(256, 256))
    obs, _ = env.reset(seed=seed)
    kin = Kinematics()
    sk = Skills(IKController(kin), kin)
    sk.tick(obs, env.objects())            # sets internal joint/tip caches before handle()
    sk.handle(Intent("pick", obj=color), color)
    stage = "pick"
    rec = {k: [] for k in ("state", "prev_goal", "action", "wrist", "front", "phase")}
    for k in range(int(max_s * 30)):
        wrist = obs["wrist"]
        front = env.sim.render("demo", 256, 256)
        state, pg = np.asarray(obs["joints"], np.float32), np.asarray(obs["prev_goal"], np.float32)
        a = sk.tick(obs, env.objects())
        phase = sk.status.phase
        obs, *_ = env.step(a)
        rec["state"].append(state); rec["prev_goal"].append(pg)
        rec["action"].append(np.asarray(obs["prev_goal"], np.float32))
        rec["wrist"].append(wrist); rec["front"].append(front); rec["phase"].append(phase)
        if sk.gen is None:
            if stage == "pick":
                if sk.status.holding != color:
                    break
                sk.handle(Intent("place", where="bowl"), color)
                stage = "place"
            elif stage == "place":
                stage = "done"
                tail = 6                              # a few frames of the final hover
            elif stage == "done":
                tail -= 1
                if tail <= 0:
                    break
    objs = env.objects()
    ok = stage == "done" and in_bowl(objs, color)
    env.close()
    n = len(rec["state"])
    if not ok:
        return {"seed": seed, "ok": False, "n": n, "color": color}
    st, act = np.stack(rec["state"]), np.stack(rec["action"])
    # trim static dwells (no goal change and no joint motion): keep the first 2 frames of every static run
    dg = np.abs(np.diff(np.r_[act[:1] * 0 + rec["prev_goal"][0], act], axis=0)).max(1)
    dj = np.abs(np.diff(np.r_[st[:1], st], axis=0)).max(1)
    static = (dg < 0.05) & (dj < 0.35)
    keep, run = np.ones(n, bool), 0
    for i in range(n):
        run = run + 1 if static[i] else 0
        if run > 2:
            keep[i] = False
    idx = np.nonzero(keep)[0]
    wj = [_jpg(rec["wrist"][i]) for i in idx]
    fj = [_jpg(rec["front"][i]) for i in idx]
    np.savez(Path(out_dir) / f"ep_{seed:06d}.npz",
             state=st[idx], prev_goal=np.stack(rec["prev_goal"])[idx], action=act[idx],
             wrist_jpg=np.concatenate(wj), wrist_off=np.cumsum([0] + [len(x) for x in wj]),
             front_jpg=np.concatenate(fj), front_off=np.cumsum([0] + [len(x) for x in fj]),
             phase=np.array([rec["phase"][i] for i in idx]), task=np.array(task), color=np.array(color),
             seed=np.array(seed), raw_len=np.array(n))
    grip = act[idx, 5]
    return {"seed": seed, "ok": True, "n": n, "kept": len(idx), "color": color,
            "grip_min": float(grip.min()), "grip_max": float(grip.max())}


def _safe(args):
    try:
        return run_episode(*args)
    except Exception as e:  # noqa: BLE001
        return {"seed": args[0], "ok": False, "err": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/vla_demos")
    ap.add_argument("--seeds", default="10000:10700")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    todo = [(s, a.out) for s in range(lo, hi) if not (Path(a.out) / f"ep_{s:06d}.npz").exists()]
    t0, ok, done = time.time(), 0, 0
    with Pool(a.workers) as p:
        for r in p.imap_unordered(_safe, todo):
            done += 1; ok += r["ok"]
            if done % 10 == 0 or "err" in r:
                print(f"{time.time() - t0:6.0f}s {done}/{len(todo)} ok {ok} last {r}", flush=True)
    print(f"DONE {ok}/{len(todo)} successful in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
