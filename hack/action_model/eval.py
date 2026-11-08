"""Evaluate the exported numpy ActionModel (what the app runs) under DR, on four conditions x 50 episodes.

  .venv/Scripts/python.exe -m hack.action_model.eval [--episodes 50] [--dr 1.0] [--baseline]
--baseline also runs the damped-least-squares IK controller of examples/scripted_reach.py (gain 0.3, nominal
model, no hover phase, ignores `down`) on the same episodes.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from so101_sim2real.contract import CONTROL_HZ
from so101_sim2real.env import DRConfig

from .model import ActionModel
from .train_env import ReachEnv

CONDS = {
    "static_down": dict(down=1, redirect=0, drift=0),
    "static_free": dict(down=0, redirect=0, drift=0),
    "redirect": dict(redirect=1, drift=0),
    "drift": dict(redirect=0, drift=1),
}


class DLS:
    def __init__(self, kin):
        self.kin = kin

    def act(self, joints, prev_goal, target, down):
        q = np.asarray(joints, float)
        tip, _ = self.kin.fk(q)
        J = np.zeros((3, 4)); h = 1e-4
        for i in range(4):
            dq = q.copy(); dq[i] += np.rad2deg(h)
            J[:, i] = (self.kin.fk(dq)[0] - tip) / h
        e = np.clip(0.3 * (np.asarray(target) - tip), -0.01, 0.01)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
        a = np.zeros(6); a[:4] = np.clip(np.rad2deg(dq) / 6.0, -1, 1)
        return a


def run(policy, env: ReachEnv, cond: dict, episodes: int, seed: int) -> dict:
    env.forced = cond
    fin, succ, align_deg, lag, never, reach_t, track, contacts = [], [], [], [], 0, [], [], 0
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        policy.reset() if hasattr(policy, "reset") else None
        dists, done, first = [], False, None
        while not done:
            a = policy.act(env.joints, env.goal, env.target, float(env.down))
            o, r, te, tr, info = env.step(a[:4])
            dists.append(info["dist"]); contacts += info["contact"]
            if first is None and info["success"]:
                first = info["t"]
            done = te or tr
        fin.append(info["dist"]); succ.append(info["success"])
        if env.down:
            align_deg.append(np.degrees(np.arccos(np.clip(info["align"], -1, 1))))
        if first is not None and env.redirect_at < 0:
            reach_t.append(first / CONTROL_HZ)
        if env.redirect_at > 0:
            d = np.array(dists[env.redirect_at:])
            hit = np.flatnonzero(d < 0.015)
            if len(hit):
                lag.append(hit[0] / CONTROL_HZ)
            else:
                never += 1
        track.append(np.mean(dists[-60:]))
    fin = np.array(fin) * 1000
    out = {"median_mm": round(float(np.median(fin)), 1), "p90_mm": round(float(np.percentile(fin, 90)), 1),
           "success": f"{int(np.sum(succ))}/{episodes}",
           "track_last2s_mm": round(float(np.mean(track)) * 1000, 1), "contact_steps": int(contacts)}
    if align_deg:
        out["down_err_deg_median"] = round(float(np.median(align_deg)), 1)
        out["down_err_deg_p90"] = round(float(np.percentile(align_deg, 90)), 1)
    if reach_t:
        out["time_to_15mm_s_median"] = round(float(np.median(reach_t)), 2)
    if lag or never:
        out["redirect_lag_s_median"] = round(float(np.median(lag)), 2) if lag else None
        out["redirect_lag_s_p90"] = round(float(np.percentile(lag, 90)), 2) if lag else None
        out["redirect_never_reached"] = never
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--dr", type=float, default=1.0)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--path", default=None)
    ap.add_argument("--no-calib-err", action="store_true", help="full DR except the joint-zero calibration error")
    a = ap.parse_args()
    dr = DRConfig() if a.dr == 1.0 else (DRConfig.off() if a.dr == 0 else DRConfig(scale=a.dr))
    if a.no_calib_err:
        dr.offset_sigma_deg = (0.0,) * 6
    env = ReachEnv(dr=dr, seed=0)
    pols = {"policy": ActionModel(a.path) if a.path else ActionModel()}
    if a.baseline:
        pols["dls_ik"] = DLS(env.kin)
    res = {}
    for pname, pol in pols.items():
        for cname, cond in CONDS.items():
            r = run(pol, env, cond, a.episodes, seed=50_000)
            res[f"{pname}/{cname}"] = r
            print(f"dr={a.dr} {pname:8s} {cname:12s} {json.dumps(r)}", flush=True)
    return res


if __name__ == "__main__":
    main()
