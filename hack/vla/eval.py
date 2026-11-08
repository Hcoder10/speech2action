"""Closed-loop eval of the VLA skill in DemoEnv on held-out layouts (seeds never used for demos: demos 10000-10699).

  python -m hack.vla.eval --n 20   (server: hack/vla/server.py running) --replan 5 --ens 0.01 [--video runs/vla_eval.mp4]
Success = the named cube ends in the bowl (xy within 6 cm of the bowl centre, z < 6 cm).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

COLORS = ("red", "blue", "green")
TEMPLATES = ("pick up the {c} cube and put it in the bowl", "put the {c} cube in the bowl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--ens", type=float, default=0.01)
    ap.add_argument("--max_s", type=float, default=20.0)
    ap.add_argument("--video")
    ap.add_argument("--out")
    a = ap.parse_args()
    from hack.scene import DemoEnv
    from hack.vla.gen_demos import in_bowl
    from hack.vla.skill import VLAClient, VLASkill

    pol = VLAClient()
    rows, frames, lat = [], [], []
    for i in range(a.n):
        seed, color = a.seed0 + i, COLORS[i % 3]
        instr = TEMPLATES[(i // 3) % 2].format(c=color)
        env = DemoEnv(seed=seed, obs_keys=("joints", "prev_goal", "wrist"), image_size=(256, 256))
        obs, _ = env.reset(seed=seed)
        run = VLASkill(pol, instr, replan=a.replan, ens_k=a.ens)
        t_succ, hold, wrong, t0 = None, 0, None, time.time()
        for k in range(int(a.max_s * 30)):
            wrist, front = obs["wrist"], env.sim.render("demo", 256, 256)
            obs, *_ = env.step(run.act(obs, wrist, front))
            if a.video and i < 6 and k % 2 == 0:
                frames.append(env.render_view(w=480, h=360))
            o = env.objects()
            ok = in_bowl(o, color) and obs["prev_goal"][5] > 25       # in the bowl and released
            hold = hold + 1 if ok else 0
            if ok and t_succ is None:
                t_succ = (k + 1) / 30
            if hold >= 15:
                break
        o = env.objects()
        others = [c for c in COLORS if c != color and in_bowl(o, c)]
        succ = in_bowl(o, color)
        lat += run.latency
        rows.append({"seed": seed, "color": color, "instr": instr, "success": bool(succ), "t": t_succ,
                     "wrong_in_bowl": others, "wall": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
        env.close()
    s = sum(r["success"] for r in rows)
    ts = [r["t"] for r in rows if r["success"] and r["t"]]
    by = {c: f'{sum(r["success"] for r in rows if r["color"] == c)}/{sum(r["color"] == c for r in rows)}' for c in COLORS}
    summ = {"replan": a.replan, "ens": a.ens, "success": f"{s}/{len(rows)}", "by_color": by,
            "median_t": float(np.median(ts)) if ts else None,
            "latency_ms_median": 1000 * float(np.median(lat)), "latency_ms_p90": 1000 * float(np.percentile(lat, 90))}
    print("SUMMARY", json.dumps(summ), flush=True)
    if a.out:
        json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    if a.video and frames:
        import cv2
        w = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (480, 360))
        for f in frames:
            w.write(f[..., ::-1])
        w.release()


if __name__ == "__main__":
    main()
