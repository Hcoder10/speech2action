"""Fit the STS3215 model to a recording.

  python -m so101_sim2real.sysid.fit runs/rec_A.npz --out so101_sim2real/assets/fitted.json [--holdout runs/rec_B.npz]

Stages (each joint's segment moves only that joint, so per-joint fits are nearly independent):
  A  latency + shoulder_pan together (pan carries no gravity load, so its phase lag is mostly latency)
  B  every other joint, latency fixed, in parallel processes; the gravity segment joins the pitch joints
  B2 B repeated once with the other joints at their stage-B values (coupling through gravity and inertia)
  C  latency re-fitted on everything with all joint params fixed
Positive params are fitted in log space with bounds. Residuals are degrees (gripper units converted to degrees).
The covariance from the Jacobian is inflated for the autocorrelation of the residuals (effective sample rate
`eff_hz`), and becomes the domain-randomisation box via ParamRanges.around (floor +-15 %).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from ..contract import JOINTS
from ..servo import ParamRanges, ServoParams, prior
from .log import Recording
from .replay import make_sim, replay_segment

FIT = ("kp", "kv", "tau", "omega_free", "frictionloss", "armature")
LO = {"kp": 5.0, "kv": 1e-3, "tau": 0.2, "omega_free": 0.5, "frictionloss": 1e-4, "armature": 1e-4}
HI = {"kp": 5e3, "kv": 50.0, "tau": 10.0, "omega_free": 30.0, "frictionloss": 1.0, "armature": 0.2}
OTHER_WEIGHT = 0.3
OBS_HZ = 60.0


def _scale(sim):
    return np.r_[np.ones(5), sim.contract.gripper_deg_per_unit]


def _segments_for(rec: Recording, j: int) -> list[dict]:
    segs = [s for s in rec.segments if s["joint"] == j]
    if j in (1, 2, 3):
        segs += [s for s in rec.segments if s["name"] == "gravity"]
    return segs


def residuals(p: ServoParams, rec: Recording, segs: list[dict], j: int | None, sim=None) -> np.ndarray:
    sim = sim or make_sim(p)
    sim.set_params(p)
    w = np.full(6, OTHER_WEIGHT) if j is not None else np.ones(6)
    if j is not None:
        w[j] = 1.0
    out = []
    for seg in segs:
        oi, pred = replay_segment(sim, rec, seg, max_obs_hz=OBS_HZ)
        out.append(((pred - rec.obs[oi]) * _scale(sim) * w).ravel())
    return np.concatenate(out)


def _with(p: ServoParams, j: int, x: np.ndarray, latency: float | None = None) -> ServoParams:
    q = p.copy()
    for k, name in enumerate(FIT):
        arr = getattr(q, name).copy(); arr[j] = np.exp(x[k]); setattr(q, name, arr)
    if latency is not None:
        q.latency = float(latency)
    return q


def fit_joint(args) -> dict:
    p, rec, j, with_latency, max_nfev = args
    segs = _segments_for(rec, j)
    sim = make_sim(p)
    x0 = np.log([getattr(p, n)[j] for n in FIT])
    lo = np.log([LO[n] for n in FIT]); hi = np.log([HI[n] for n in FIT])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    if with_latency:
        x0 = np.r_[x0, p.latency * 100]; lo = np.r_[lo, 0.0]; hi = np.r_[hi, 12.0]   # latency in 10 ms units

    def f(x):
        q = _with(p, j, x[:len(FIT)], x[len(FIT)] / 100 if with_latency else None)
        return residuals(q, rec, segs, j, sim)

    r0 = f(x0)
    # coarse multi-start on (kp, tau): saturation makes the cost non-convex in these two
    best = (np.sqrt(np.mean(r0 ** 2)), x0)
    for dkp in (-1.5, 0.0):
        for dtau in (-0.4, 0.0):
            x = x0.copy(); x[0] = np.clip(x0[0] + dkp, lo[0], hi[0]); x[2] = np.clip(x0[2] + dtau, lo[2], hi[2])
            c = np.sqrt(np.mean(f(x) ** 2))
            if c < best[0]:
                best = (c, x)
    sol = least_squares(f, best[1], bounds=(lo, hi), loss="soft_l1", f_scale=0.5, diff_step=0.03,
                        x_scale=1.0, max_nfev=max_nfev)
    J = sol.jac
    n, k = J.shape
    s2 = float(np.sum(sol.fun ** 2) / max(1, n - k))
    try:
        cov = np.linalg.pinv(J.T @ J) * s2 * (OBS_HZ / 10.0)   # ~10 Hz of independent information
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.diag(np.full(k, 0.25)); sig = np.full(k, 0.5)
    return {"joint": j, "x": sol.x.tolist(), "sigma": sig.tolist(), "cov": cov[:len(FIT), :len(FIT)].tolist(),
            "rmse_before": float(np.sqrt(np.mean(r0 ** 2))), "rmse_after": float(np.sqrt(np.mean(sol.fun ** 2))),
            "nfev": int(sol.nfev)}


def fit(rec: Recording, start: ServoParams | None = None, workers: int = 6, max_nfev: int = 60,
        joints=range(6), log=print) -> tuple[ServoParams, ParamRanges, dict]:
    p = (start or prior()).copy()
    report = {"stages": []}
    sig = {n: np.zeros(6) for n in FIT}
    covs = [np.diag(np.full(len(FIT), 0.25)) for _ in range(6)]      # unfitted joints: +-50 % (1 sigma)
    lat_sigma = 0.0

    t = time.time()
    if 0 in joints:
        r = fit_joint((p, rec, 0, True, max_nfev))
        p = _with(p, 0, np.array(r["x"][:len(FIT)]), r["x"][len(FIT)] / 100)
        for k, n in enumerate(FIT): sig[n][0] = r["sigma"][k]
        covs[0] = np.array(r["cov"])
        lat_sigma = r["sigma"][len(FIT)] / 100
        report["stages"].append({"stage": "A", **{k: v for k, v in r.items() if k != "cov"}})
        log(f"A pan+latency: rmse {r['rmse_before']:.3f} -> {r['rmse_after']:.3f} deg, latency {p.latency * 1000:.1f} ms ({time.time() - t:.0f}s)")

    rest = [j for j in joints if j != 0]
    for stage in ("B", "B2"):
        t = time.time()
        with ProcessPoolExecutor(max_workers=min(workers, len(rest))) as ex:
            results = list(ex.map(fit_joint, [(p, rec, j, False, max_nfev) for j in rest]))
        for r in results:
            j = r["joint"]
            p = _with(p, j, np.array(r["x"]))
            for k, n in enumerate(FIT): sig[n][j] = r["sigma"][k]
            covs[j] = np.array(r["cov"])
            report["stages"].append({"stage": stage, **{k: v for k, v in r.items() if k != "cov"}})
            log(f"{stage} {JOINTS[j]:>13}: rmse {r['rmse_before']:.3f} -> {r['rmse_after']:.3f} deg")
        log(f"{stage} done ({time.time() - t:.0f}s)")

    t = time.time()
    segs = rec.segments
    sim = make_sim(p)

    def cost(lat):
        q = p.copy(); q.latency = float(lat)
        return float(np.mean(residuals(q, rec, segs, None, sim) ** 2))
    res = minimize_scalar(cost, bounds=(max(0.0, p.latency - 0.03), p.latency + 0.03), method="bounded",
                          options={"xatol": 5e-4})
    p.latency = float(res.x)
    report["stages"].append({"stage": "C", "latency": p.latency, "rmse": float(np.sqrt(res.fun))})
    log(f"C latency {p.latency * 1000:.1f} ms, overall rmse {np.sqrt(res.fun):.3f} deg ({time.time() - t:.0f}s)")

    p.source = f"fitted: {rec.meta.get('source', 'recording')}"
    ranges = ParamRanges.around(p, sigma={**sig, "damping": np.zeros(6)}, latency_sigma=lat_sigma)
    ranges.cov = [c.tolist() for c in covs]
    report["sigma_log"] = {k: v.tolist() for k, v in sig.items()}
    return p, ranges, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "assets" / "fitted.json"))
    ap.add_argument("--holdout", help="second recording for the validation report")
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rec = Recording.load_npz(Path(a.recording))
    p, ranges, report = fit(rec, workers=a.workers, max_nfev=a.max_nfev)
    out = {"params": p.to_json(), "ranges": ranges.to_json(), "fit": report,
           "recording": str(Path(a.recording).resolve()), "recording_meta": rec.meta,
           "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.holdout:
        from .report import compare
        out["holdout"] = compare(Recording.load_npz(Path(a.holdout)), {"fitted": p, "prior": prior()})
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)
    if a.holdout:
        from .report import print_table
        print_table(out["holdout"])


if __name__ == "__main__":
    main()
