"""Fine-tune SmolVLA on the expert demos (hack/vla/gen_demos.py). Owner joint frame, ABSOLUTE joint-goal actions,
fresh mean/std normalizers from the demos. Cameras: camera1 = fixed front view ("demo"), camera2 = wrist (same slot
roles as the canonical SO-101 run: camera1 top/scene, camera2 wrist).

Run with the LeRobot 0.6.1 venv:
  HF_HOME=runs/hf C:/Users/sarta/reachy-duplex/.venv-so101/Scripts/python.exe -m hack.vla.train --minutes 40
Saves only the trainable weights (+ stats + config) to --out; the frozen VLM comes from the init checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

INIT = "C:/Users/sarta/ego16-storage/so101/ckpt/canon_run_best/step_0020000"
CHUNK = 50
TOK_KEY, MASK_KEY = "observation.language.tokens", "observation.language.attention_mask"


class Demos:
    def __init__(self, root, holdout_frac=0.0):
        self.eps = []
        for f in sorted(glob.glob(str(Path(root) / "ep_*.npz"))):
            d = np.load(f)
            self.eps.append({k: d[k] for k in ("state", "action", "wrist_jpg", "wrist_off", "front_jpg", "front_off")}
                            | {"task": str(d["task"])})
        self.index = np.array([(e, t) for e, ep in enumerate(self.eps) for t in range(len(ep["state"]))])
        st = np.concatenate([e["state"] for e in self.eps]); ac = np.concatenate([e["action"] for e in self.eps])
        self.stats = {"state_mean": st.mean(0), "state_std": np.maximum(st.std(0), 1e-2),
                      "action_mean": ac.mean(0), "action_std": np.maximum(ac.std(0), 1e-2)}

    @staticmethod
    def _img(ep, key, t):
        o = ep[key + "_off"]
        return cv2.imdecode(ep[key + "_jpg"][o[t]:o[t + 1]], cv2.IMREAD_COLOR)[..., ::-1]

    state_noise = None          # per-joint sigma (deg): forces the policy to read the target from pixels, not momentum

    def set_early_weight(self, w, t_max=40):
        p = np.where(self.index[:, 1] < t_max, w, 1.0)
        self.p = p / p.sum()

    def sample(self, rng, bs):
        p = getattr(self, "p", None)
        idx = self.index[rng.choice(len(self.index), size=bs, p=p) if p is not None else rng.integers(len(self.index), size=bs)]
        front, wrist, state, act, pad, tasks = [], [], [], [], [], []
        for e, t in idx:
            ep = self.eps[e]
            n = len(ep["state"])
            front.append(self._img(ep, "front", t)); wrist.append(self._img(ep, "wrist", t))
            state.append(ep["state"][t] + (rng.normal(0, 1, 6) * self.state_noise if self.state_noise is not None else 0))
            j = np.minimum(np.arange(t, t + CHUNK), n - 1)
            act.append(ep["action"][j]); pad.append(np.arange(t, t + CHUNK) >= n)
            tasks.append(ep["task"])
        s = self.stats
        return {
            "front": np.stack(front), "wrist": np.stack(wrist),
            "state": (np.stack(state) - s["state_mean"]) / s["state_std"],
            "action": (np.stack(act) - s["action_mean"]) / s["action_std"],
            "pad": np.stack(pad), "tasks": tasks,
        }


def build_policy(device, init=INIT):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = SmolVLAConfig.from_pretrained(init)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    cfg.empty_cameras = 0
    cfg.device = device
    pol = SmolVLAPolicy.from_pretrained(init, config=cfg)
    return pol.to(device), cfg


class Tok:
    def __init__(self, name, max_len):
        from transformers import AutoTokenizer
        self.tok, self.max_len, self.cache = AutoTokenizer.from_pretrained(name), max_len, {}

    def __call__(self, tasks, device):
        out = []
        for t in tasks:
            if t not in self.cache:
                r = self.tok(t + "\n", padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt")
                self.cache[t] = (r["input_ids"][0], r["attention_mask"][0].bool())
            out.append(self.cache[t])
        return torch.stack([o[0] for o in out]).to(device), torch.stack([o[1] for o in out]).to(device)


def to_batch(b, tok, device):
    img = lambda x: torch.from_numpy(x).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    toks, mask = tok(b["tasks"], device)
    return {
        "observation.images.camera1": img(b["front"]), "observation.images.camera2": img(b["wrist"]),
        "observation.state": torch.from_numpy(b["state"]).float().to(device),
        "action": torch.from_numpy(b["action"]).float().to(device),
        "action_is_pad": torch.from_numpy(b["pad"]).to(device),
        TOK_KEY: toks, MASK_KEY: mask,
    }


def save(pol, cfg, stats, out, step, extra):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().cpu().contiguous() for k, v in pol.named_parameters() if v.requires_grad}
    tmp = out / "trainable.safetensors.tmp"
    save_file(sd, str(tmp)); os.replace(tmp, out / "trainable.safetensors")
    (out / "stats.json").write_text(json.dumps({k: v.tolist() for k, v in stats.items()}))
    (out / "meta.json").write_text(json.dumps({"init": INIT, "step": step, "frame": "owner (so_follower degrees, gripper 0..100)",
                                               "action": "absolute joint goal (obs['prev_goal'] after the step)",
                                               "camera1": "front 'demo' cam 256x256", "camera2": "wrist, contract.preprocess_image 256x256",
                                               "chunk": CHUNK} | extra, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/vla_demos")
    ap.add_argument("--out", default="runs/vla_ckpt")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--resume", help="dir with trainable.safetensors to start from")
    ap.add_argument("--early_weight", type=float, default=1.0, help="oversample the first 40 frames (approach: grounding)")
    ap.add_argument("--warm", type=int, default=150)
    ap.add_argument("--state_noise", type=float, default=0.0, help="scale on sigma (pan 6, lift/elbow/flex 4, roll 6, grip 3 deg)")
    a = ap.parse_args()
    lease_frac = float(os.environ.get("LANDLORD_VRAM_GB", 0)) / 15.92
    device = "cuda"
    if lease_frac > 0:
        torch.cuda.set_per_process_memory_fraction(min(lease_frac, 0.95))
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    data = Demos(a.data)
    print(f"{len(data.eps)} episodes, {len(data.index)} frames, load {time.time() - t0:.0f}s", flush=True)
    if a.state_noise > 0:
        data.state_noise = a.state_noise * np.array([6.0, 4.0, 4.0, 4.0, 6.0, 3.0])
    if a.early_weight != 1.0:
        data.set_early_weight(a.early_weight)
    pol, cfg = build_policy(device)
    if a.resume:
        from safetensors.torch import load_file
        _, unexpected = pol.load_state_dict(load_file(str(Path(a.resume) / "trainable.safetensors")), strict=False)
        assert not unexpected
        print("resumed from", a.resume, json.loads((Path(a.resume) / "meta.json").read_text()).get("step"), flush=True)
    pol.train()
    tok = Tok(cfg.vlm_model_name, cfg.tokenizer_max_length)
    params = [p for p in pol.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M of {sum(p.numel() for p in pol.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-10, eps=1e-8)
    budget = a.minutes * 60
    q: queue.Queue = queue.Queue(maxsize=6)

    def producer(seed):
        rng = np.random.default_rng(seed)
        while True:
            q.put(data.sample(rng, a.bs))

    for s in range(3):
        threading.Thread(target=producer, args=(s,), daemon=True).start()
    t_start, step, ema = time.time(), 0, None
    warm = a.warm
    while step < a.max_steps:
        el = time.time() - t_start
        if el > budget:
            break
        frac = min(1.0, el / budget)
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = to_batch(q.get(), tok, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = pol.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 25 == 0:
            print(f"step {step} loss {ema:.4f} lr {lr:.2e} {el / 60:.1f}min {step * a.bs / max(el, 1):.1f} samp/s "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % a.save_every == 0:
            save(pol, cfg, data.stats, a.out, step, {"loss_ema": ema, "episodes": len(data.eps)})
    save(pol, cfg, data.stats, a.out, step, {"loss_ema": ema, "episodes": len(data.eps), "final": True,
                                            "minutes": (time.time() - t_start) / 60})
    print(f"DONE step {step} loss {ema:.4f} in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
