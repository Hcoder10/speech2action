"""Hidden-state intent probe: PersonaPlex's 4096-d transformer state (12.5 Hz) -> what the user just asked for.

The model's hidden state already encodes the user's request while it is still deciding what to say, so a linear
probe can act on it before the words "grabbing the red cube" come out. Training labels come from the scripted user
audio (hack.hidden.collect): frames from just before the user finishes an utterance to ~1 s after carry its label,
frames inside the utterance are ignored, everything else is "none".

  python -m hack.hidden.probe train            # fits on runs/hidden/ep_*.npz, writes hack/hidden/probe.npz, evaluates
At runtime: HiddenProbe().push(h) -> list[Intent] (fires on a confident, sustained prediction).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ..intent import Intent, IntentParser

HERE = Path(__file__).resolve().parent
DATA = Path("runs/hidden")
CLASSES = ["none", "pick_red", "pick_green", "pick_blue", "place_bowl", "stop", "wave", "nudge_left", "nudge_right",
           "nudge_up", "cleanup", "tower", "home"]
PRE, POST = 2, 14            # label window around the end of a user utterance, in frames (80 ms)
CTX = 4                      # frames averaged with the current one


def feats(H: np.ndarray) -> np.ndarray:
    """(T, 4096) -> (T, 8192): the current frame and the mean of the previous CTX frames."""
    H = H.astype(np.float32)
    c = np.cumsum(np.vstack([np.zeros((1, H.shape[1]), np.float32), H]), 0)
    idx = np.arange(H.shape[0])
    lo = np.maximum(0, idx - CTX)
    mean = (c[idx + 1] - c[lo]) / (idx + 1 - lo)[:, None]
    return np.hstack([H, mean])


def labels(T: int, sched: list) -> np.ndarray:
    y = np.zeros(T, int)                      # none
    for s in sched:
        a, b = int(s["start"]), int(round(s["end"]))
        y[a:max(a, b - PRE)] = -1                       # mid-utterance: not decidable yet, don't train on it
        if s["label"] != "none":
            y[max(0, b - PRE):min(T, b + POST)] = CLASSES.index(s["label"])
    return y


def to_intent(c: str) -> Intent | None:
    if c.startswith("pick_"):
        return Intent("pick", obj=c[5:])
    if c == "place_bowl":
        return Intent("place", where="bowl")
    if c.startswith("nudge_"):
        return Intent("nudge", dir=c[6:], amount=0.025)
    if c in ("stop", "wave", "cleanup", "tower", "home"):
        return Intent(c)
    return None


class HiddenProbe:
    def __init__(self, path=HERE / "probe.npz", threshold: float | None = None, sustain: int = 2, refractory: float = 2.0):
        z = np.load(path)
        self.mu, self.sd, self.P, self.W, self.b = z["mu"], z["sd"], z["P"], z["W"], z["b"]
        self.th = float(z["threshold"]) if threshold is None else threshold
        self.sustain, self.refractory = sustain, refractory
        self.buf: list[np.ndarray] = []
        self.run_cls, self.run_n = 0, 0
        self.last_fire: dict[str, float] = {}
        self.last_probs = np.zeros(len(CLASSES))

    def probs(self, h: np.ndarray) -> np.ndarray:
        self.buf = (self.buf + [h.astype(np.float32)])[-(CTX + 1):]
        prev = np.mean(self.buf[:-1], 0) if len(self.buf) > 1 else self.buf[-1]
        x = (np.r_[self.buf[-1], prev] - self.mu) / self.sd
        z = (x @ self.P) @ self.W + self.b
        e = np.exp(z - z.max())
        self.last_probs = e / e.sum()
        return self.last_probs

    def push(self, h: np.ndarray) -> list[Intent]:
        return self.decide(self.probs(h))

    def push_probs(self, p: np.ndarray) -> list[Intent]:
        """Probabilities computed by the server-side probe (deploy/patch_probe_stream.py)."""
        self.last_probs = np.asarray(p, float)
        return self.decide(self.last_probs)

    def decide(self, p: np.ndarray) -> list[Intent]:
        c = int(np.argmax(p))
        if c == 0 or p[c] < self.th:
            self.run_cls, self.run_n = 0, 0
            return []
        self.run_n = self.run_n + 1 if c == self.run_cls else 1
        self.run_cls = c
        name = CLASSES[c]
        now = time.perf_counter()
        if self.run_n == self.sustain and now - self.last_fire.get(name, -1e9) > self.refractory:
            self.last_fire[name] = now
            it = to_intent(name)
            if it is not None:
                it.text = f"[hidden {p[c]:.2f}]"
                return [it]
        return []


# ---------------------------------------------------------------------------------------------- training
def load_all():
    eps = []
    for f in sorted(DATA.glob("ep_*.npz")):
        z = np.load(f)
        eps.append({"name": f.stem, "H": z["hidden"], "sched": json.loads(str(z["schedule"])),
                    "text": json.loads(str(z["text"])), "tf": z["text_frames"]})
    return eps


def text_intent_frames(ep) -> list[tuple[int, str]]:
    """When the parser on the AGENT'S WORDS fired, per frame (the baseline the probe must beat)."""
    p = IntentParser(); out = []
    last_f = -100
    for f, piece in zip(ep["tf"], ep["text"]):
        if f - last_f > 8:
            out += [(last_f, str(i)) for i in p.flush()]
        for i in p.feed(piece):
            out.append((int(f), str(i)))
        last_f = int(f)
    out += [(last_f, str(i)) for i in p.flush()]
    return out


def intent_key(label: str) -> str:
    it = to_intent(label)
    return str(it) if it else "none"


def train():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    eps = load_all()
    n_test = max(2, len(eps) // 4)
    tr, te = eps[:-n_test], eps[-n_test:]
    X = np.vstack([feats(e["H"]) for e in tr]); Y = np.concatenate([labels(len(e["H"]), e["sched"]) for e in tr])
    keep = Y >= 0
    X, Y = X[keep], Y[keep]
    mu, sd = X.mean(0), X.std(0) + 1e-3
    Xs = (X - mu) / sd
    pca = PCA(256, random_state=0).fit(Xs[::2])
    Z = pca.transform(Xs)
    w = np.ones(len(Y)); w[Y == 0] = 0.25                  # none dominates the frames
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(Z, Y, sample_weight=w)
    full_W = np.zeros((256, len(CLASSES))); full_b = np.full(len(CLASSES), -30.0)
    for k, c in enumerate(clf.classes_):
        full_W[:, c] = clf.coef_[k]; full_b[c] = clf.intercept_[k]
    out = {"mu": mu, "sd": sd, "P": pca.components_.T.astype(np.float32), "W": full_W.astype(np.float32), "b": full_b}
    print(f"train: {len(tr)} episodes, {len(Y)} frames; test: {len(te)} episodes", flush=True)

    best = None
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        np.savez(HERE / "probe.npz", threshold=th, **out)
        r = evaluate(te, th)
        print(f"th {th}: recall {r['hits']}/{r['n']}  wrong-class {r['wrong']}  false/min {r['fp_per_min']:.2f}  "
              f"median lead vs agent words {r['lead_med']:+.2f}s  median latency after user {r['lat_med']:+.2f}s", flush=True)
        score = r["hits"] - 2 * r["wrong"] - 3 * r["fp_per_min"]
        if best is None or score > best[0]:
            best = (score, th, r)
    np.savez(HERE / "probe.npz", threshold=best[1], **out)
    print("chosen threshold", best[1], json.dumps({k: v for k, v in best[2].items() if k != "rows"}))
    for row in best[2]["rows"]:
        print("  ", row)


def evaluate(eps, th):
    hits = wrong = n = fps = 0
    leads, lats, rows = [], [], []
    minutes = sum(len(e["H"]) for e in eps) / 12.5 / 60
    for e in eps:
        pr = HiddenProbe(HERE / "probe.npz", threshold=th, refractory=0)
        fires = []
        for f, h in enumerate(e["H"]):
            pr.last_fire.clear()
            for it in pr.push(h):
                fires.append((f, str(it)))
        tfires = text_intent_frames(e)
        used = set()
        for s in e["sched"]:
            if s["label"] == "none":
                continue
            n += 1
            end = s["end"]
            want = intent_key(s["label"])
            win = [(f, k) for f, k in fires if end - PRE <= f <= end + 25]
            ok = [(f, k) for f, k in win if k == want]
            twin = [f for f, k in tfires if end - PRE <= f <= end + 40 and k == want]
            if ok:
                hits += 1; used.add(ok[0])
                lats.append((ok[0][0] - end) / 12.5)
                if twin:
                    leads.append((twin[0] - ok[0][0]) / 12.5)
            elif win:
                wrong += 1
            rows.append(f"{e['name']} {s['label']:<11} probe {'@%+.2fs' % ((ok[0][0]-end)/12.5) if ok else ('WRONG ' + win[0][1] if win else 'miss')}"
                        f"  words {'@%+.2fs' % ((twin[0]-end)/12.5) if twin else 'never'}")
        sched_wins = [(s["end"] - PRE, s["end"] + 25) for s in e["sched"] if s["label"] != "none"]
        fps += sum(1 for f, k in fires if not any(a <= f <= b for a, b in sched_wins))
    return {"hits": hits, "wrong": wrong, "n": n, "fp_per_min": fps / max(minutes, 1e-6),
            "lead_med": float(np.median(leads)) if leads else float("nan"),
            "lat_med": float(np.median(lats)) if lats else float("nan"), "rows": rows}


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
