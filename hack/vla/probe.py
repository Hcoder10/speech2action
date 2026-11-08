"""Offline grounding probe on held-out expert episodes: does the policy aim at the NAMED cube?

At t=0 and t=10 of each held-out episode, predict a chunk and compare the predicted shoulder_pan at the expert's
grasp tick with the expert's grasp pan. Baseline: always predict the training-mean grasp pan. Also re-asks with a
WRONG colour to see whether language moves the target.
  python -m hack.vla.probe --data runs/vla_holdout      (server running)
"""
from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np

from hack.vla.skill import VLAClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/vla_holdout")
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    c = VLAClient(("127.0.0.1", a.port))
    errs, base, wrong_shift, e0 = [], [], [], []
    for f in sorted(glob.glob(f"{a.data}/ep_*.npz")):
        d = np.load(f)
        img = lambda k, i: cv2.imdecode(d[k + "_jpg"][d[k + "_off"][i]:d[k + "_off"][i + 1]], cv2.IMREAD_COLOR)[..., ::-1]
        g = int(np.argmax(np.asarray(d["phase"]) == "grasp"))
        pan_true = d["action"][g, 0]
        task, color = str(d["task"]), str(d["color"])
        for t in (0, 10):
            ch = c.chunk(d["state"][t], img("wrist", t), img("front", t), task)
            k = min(g - t, 49)
            errs.append(abs(ch[k, 0] - pan_true)); base.append(abs(0.0 - pan_true))
            if t == 0:
                e0.append(abs(ch[k, 0] - pan_true)); p0 = ch[k, 0]
            if t == 0:
                other = next(x for x in ("red", "blue", "green") if x != color)
                ch2 = c.chunk(d["state"][t], img("wrist", t), img("front", t), task.replace(color, other))
                wrong_shift.append(abs(ch2[k, 0] - ch[k, 0]))
        print(f"{f[-10:-4]} {color:5s} grasp pan {pan_true:6.1f}  pred@t0 {p0:6.1f} pred@t10 {ch[k, 0]:6.1f} pan state t10 {d['state'][10][0]:6.1f}", flush=True)
    print(f"PROBE pan error at grasp: from t=0 median {np.median(e0):.1f}, overall median {np.median(errs):.1f} deg, zero-pan baseline {np.median(base):.1f} deg; "
          f"swapping the colour word moves the predicted pan by median {np.median(wrong_shift):.1f} deg", flush=True)


if __name__ == "__main__":
    main()
