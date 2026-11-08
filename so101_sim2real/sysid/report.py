"""Sim-vs-real gap on a held-out recording.

For each candidate model, replay every segment and measure the per-joint position error against what the real
arm read. `ideal` is the baseline every sim implicitly assumes when it is not identified: the joint is wherever
the last goal said, instantly. A model earns trust only by beating it by a wide margin.

  python -m so101_sim2real.sysid.report runs/rec_B.npz [--params so101_sim2real/assets/fitted.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..contract import JOINTS
from ..servo import ServoParams, load_params, prior
from .log import Recording
from .replay import make_sim, replay_segment

GATE_VS_IDEAL = 0.5     # fitted RMSE must be at most half of ideal-tracking RMSE on every excited joint


def _ideal(rec: Recording, oi: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(rec.cmd_t, rec.obs_t[oi], side="right") - 1
    pred = rec.cmd[np.clip(idx, 0, None)].copy()
    pred[idx < 0] = rec.obs[oi][idx < 0]
    return pred


def compare(rec: Recording, models: dict[str, ServoParams]) -> dict:
    out = {name: {"rmse": np.zeros(6), "p95": np.zeros(6), "n": np.zeros(6)} for name in [*models, "ideal"]}
    errs = {name: [[] for _ in range(6)] for name in out}
    for name, p in models.items():
        sim = make_sim(p)
        for seg in rec.segments:
            oi, pred = replay_segment(sim, rec, seg)
            js = range(6) if seg["joint"] < 0 else [seg["joint"]]
            for j in js:
                errs[name][j].append(pred[:, j] - rec.obs[oi, j])
            if name == next(iter(models)):
                ideal = _ideal(rec, oi)
                for j in js:
                    errs["ideal"][j].append(ideal[:, j] - rec.obs[oi, j])
    table = {}
    for name in out:
        table[name] = {}
        for j in range(6):
            e = np.concatenate(errs[name][j]) if errs[name][j] else np.array([np.nan])
            table[name][JOINTS[j]] = {"rmse": float(np.sqrt(np.nanmean(e ** 2))), "p95": float(np.nanpercentile(np.abs(e), 95))}
    gate = {}
    if "fitted" in table:
        for j in JOINTS:
            f, i = table["fitted"][j]["rmse"], table["ideal"][j]["rmse"]
            gate[j] = bool(f <= GATE_VS_IDEAL * i)
    return {"table": table, "gate": gate, "gate_rule": f"fitted rmse <= {GATE_VS_IDEAL} x ideal-tracking rmse",
            "units": "degrees (gripper: 0-100 units)"}


def print_table(res: dict) -> None:
    names = list(res["table"])
    print(f"{'joint':>14} " + " ".join(f"{n:>16}" for n in names) + "   gate")
    for j in JOINTS:
        cells = " ".join(f"{res['table'][n][j]['rmse']:7.3f} ({res['table'][n][j]['p95']:5.2f})" for n in names)
        g = res["gate"].get(j)
        print(f"{j:>14} {cells}   {'PASS' if g else ('FAIL' if g is not None else '')}")
    print("cells: rmse (p95 |error|), " + res["units"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--params", default="fitted")
    a = ap.parse_args()
    rec = Recording.load_npz(Path(a.recording))
    fitted, _ = load_params(a.params)
    res = compare(rec, {"fitted": fitted, "prior": prior()})
    print_table(res)
    print(json.dumps(res["gate"]))


if __name__ == "__main__":
    main()
