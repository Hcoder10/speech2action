"""STS3215 servo model and its parameters.

What the real servo does (Feetech STS3215, LeRobot position mode, P=16 I=0 D=32):
  duty   = clip(Kp * (goal - pos) - Kd * vel, -limit, +limit)       firmware PD on the output-shaft encoder
  torque = tau_stall * duty - (tau_stall / omega_free) * vel         DC motor: back-EMF makes speed saturate
  plus gearbox friction (Coulomb + viscous) and reflected rotor inertia, and a bus/firmware delay between the
  goal being written and the loop acting on it.

How it maps onto native MuJoCo, so it runs at full speed and on MJX unchanged:
  * `position` actuator: kp = tau_stall*Kp, kv = tau_stall*Kd, forcerange = +-tau_stall*limit   (the duty part)
  * joint damping       = tau_stall / omega_free + viscous gearbox damping   (back-EMF acts OUTSIDE the clip)
  * joint frictionloss  = Coulomb friction of the gear train
  * joint armature      = reflected rotor inertia
Only two things are not native and live in `ArmSim`: the command delay, and the gripper's overload protection
(LeRobot sets Overload_Torque 25 on the gripper: after a sustained stall the firmware drops to 25 % torque).

Parameters are stored in physical units per joint. `prior()` is the vendor model (unvalidated); `fitted.json`
from `sysid.fit` replaces it. `ranges` give the domain-randomisation box, in log space for positive quantities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .contract import JOINTS

POSITIVE = ("kp", "kv", "tau", "omega_free", "frictionloss", "damping", "armature")


@dataclass
class ServoParams:
    kp: np.ndarray            # Nm/rad, inside the clip
    kv: np.ndarray            # Nm s/rad, inside the clip
    tau: np.ndarray           # Nm, torque at full duty x torque limit
    omega_free: np.ndarray    # rad/s, no-load speed at full duty (back-EMF slope = tau_stall / omega_free)
    frictionloss: np.ndarray  # Nm
    damping: np.ndarray       # Nm s/rad, viscous gearbox damping (in addition to back-EMF)
    armature: np.ndarray      # kg m^2
    latency: float = 0.020    # s, goal write -> servo loop acts on it (minus nothing: obs are timestamped)
    gripper_overload_frac: float = 0.25
    gripper_overload_time: float = 2.0     # s at >= 95 % torque before the firmware cuts to overload torque
    source: str = "prior"
    notes: dict = field(default_factory=dict)

    def copy(self) -> "ServoParams":
        return ServoParams(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})

    def to_json(self) -> dict:
        out = {}
        for k, v in asdict(self).items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        out["joints"] = list(JOINTS)
        return out

    @classmethod
    def from_json(cls, d: dict) -> "ServoParams":
        d = dict(d)
        d.pop("joints", None)
        for k in POSITIVE:
            d[k] = np.asarray(d[k], float)
        return cls(**d)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "ServoParams":
        data = json.loads(Path(path).read_text())
        return cls.from_json(data["params"] if "params" in data else data)


def prior() -> ServoParams:
    """Vendor SO-101 model (TheRobotStudio so101_new_calib.xml, gains derived for P=16) + STS3215 datasheet.

    NOT measured on the owner's arm. 12 V STS3215: stall 2.94 Nm (30 kg cm), no-load 0.222 s / 60 deg = 4.7 rad/s.
    The gripper torque limit is halved (LeRobot writes Max_Torque_Limit 500 on it).
    """
    n = len(JOINTS)
    tau = np.full(n, 2.94)
    tau[5] = 1.47
    return ServoParams(
        kp=np.full(n, 998.22),
        kv=np.full(n, 2.731),
        tau=tau,
        omega_free=np.full(n, 4.7),
        frictionloss=np.full(n, 0.052),
        damping=np.full(n, 0.0),
        armature=np.full(n, 0.028),
        latency=0.020,
        source="prior: vendor MJCF + STS3215 datasheet, unvalidated on this arm",
    )


COV_PARAMS = ("kp", "kv", "tau", "omega_free", "frictionloss", "armature")


@dataclass
class ParamRanges:
    """Per-parameter randomisation.

    Without `cov`: positive quantities multiply by exp(U(-w, w)) independently (the prior's wide box).
    With `cov` (from a fit): per joint, log-params are drawn from the fit's posterior N(0, cov) truncated at 2 sigma,
    which keeps the correlations the data imposes (on unloaded joints kp, kv, tau and armature are only identified
    jointly; independent sampling would create servos the arm cannot have), plus an independent U(-floor, floor)
    for what the fit cannot see (temperature, supply voltage sag, wear). Latency adds U(lo, hi).
    """
    log_halfwidth: dict          # name -> array(6) of log half-widths
    latency: tuple               # (lo, hi) seconds
    cov: list | None = None      # per joint: 6x6 covariance of log COV_PARAMS, or None
    floor: float = float(np.log(1.10))

    @classmethod
    def around(cls, p: ServoParams, sigma: dict | None = None, floor: float = np.log(1.15),
               prior_width: float | None = None, latency_sigma: float = 0.0, latency_floor: float = 0.005):
        """Box = fit +- max(2 sigma, floor). With `prior_width` every positive param gets that log half-width."""
        hw = {}
        for k in POSITIVE:
            if prior_width is not None:
                hw[k] = np.full(len(JOINTS), prior_width)
            else:
                s = np.asarray((sigma or {}).get(k, np.zeros(len(JOINTS))), float)
                hw[k] = np.maximum(2.0 * s, floor)
        lw = max(2.0 * latency_sigma, latency_floor)
        return cls(log_halfwidth=hw, latency=(max(0.0, p.latency - lw), p.latency + lw))

    def sample(self, p: ServoParams, rng: np.random.Generator) -> ServoParams:
        q = p.copy()
        if self.cov is None:
            for k in POSITIVE:
                w = self.log_halfwidth[k]
                setattr(q, k, getattr(p, k) * np.exp(rng.uniform(-w, w)))
        else:
            for j, cov in enumerate(self.cov):
                cov = np.asarray(cov, float)
                w, V = np.linalg.eigh(0.5 * (cov + cov.T))
                z = rng.standard_normal(len(w))
                while np.any(np.abs(z) > 2):
                    bad = np.abs(z) > 2; z[bad] = rng.standard_normal(bad.sum())
                dx = V @ (np.sqrt(np.clip(w, 0, None)) * z)
                dx = np.clip(dx, -1.0, 1.0)     # never more than x2.7 from the fit, whatever the covariance says
                dx += rng.uniform(-self.floor, self.floor, len(dx))
                for k, name in enumerate(COV_PARAMS):
                    arr = getattr(q, name).copy(); arr[j] = getattr(p, name)[j] * np.exp(dx[k]); setattr(q, name, arr)
        q.latency = float(rng.uniform(*self.latency))
        q.source = p.source + " (randomised)"
        return q

    def to_json(self) -> dict:
        return {"log_halfwidth": {k: np.asarray(v).tolist() for k, v in self.log_halfwidth.items()},
                "latency": list(self.latency), "floor": self.floor,
                "cov": None if self.cov is None else [np.asarray(c).tolist() for c in self.cov]}

    @classmethod
    def from_json(cls, d: dict) -> "ParamRanges":
        return cls({k: np.asarray(v, float) for k, v in d["log_halfwidth"].items()}, tuple(d["latency"]),
                   d.get("cov"), d.get("floor", float(np.log(1.10))))


def load_params(spec: str | Path | ServoParams | None) -> tuple[ServoParams, ParamRanges]:
    """'prior' | 'fitted' | path to a fitted.json | a ServoParams. Returns params and their DR box."""
    if isinstance(spec, ServoParams):
        return spec, ParamRanges.around(spec)
    if spec in (None, "fitted"):
        path = Path(__file__).resolve().parent / "assets" / "fitted.json"
        if not path.exists():
            if spec == "fitted":
                raise FileNotFoundError(f"{path} missing: run sysid.fit on a real recording first")
            import warnings
            warnings.warn("no assets/fitted.json yet: using the UNVALIDATED vendor prior with wide (x2) randomisation. "
                          "Record the arm (sysid.record) and fit (sysid.fit) before trusting transfer.", stacklevel=3)
            spec = "prior"
        else:
            spec = path
    if spec == "prior":
        p = prior()
        return p, ParamRanges.around(p, prior_width=np.log(2.0), latency_floor=0.015)
    data = json.loads(Path(spec).read_text())
    p = ServoParams.from_json(data["params"])
    ranges = ParamRanges.from_json(data["ranges"]) if "ranges" in data else ParamRanges.around(p)
    return p, ranges
