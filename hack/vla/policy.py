"""SmolVLA wrapper (LeRobot 0.6.1 venv): pixels (front + wrist) + language + joints -> (50, 6) absolute joint goals.
Served by hack/vla/server.py; the app-side skill (ensembling, delta actions) is hack/vla/skill.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .train import INIT, Tok, build_policy

class VLAPolicy:
    def __init__(self, ckpt="runs/vla_ckpt", device="cuda", init=INIT):
        from safetensors.torch import load_file
        self.device = device
        self.pol, self.cfg = build_policy(device, init)
        sd = load_file(str(Path(ckpt) / "trainable.safetensors"))
        missing, unexpected = self.pol.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        self.pol.eval()
        s = json.loads((Path(ckpt) / "stats.json").read_text())
        self.s = {k: np.asarray(v, np.float32) for k, v in s.items()}
        self.tok = Tok(self.cfg.vlm_model_name, self.cfg.tokenizer_max_length)
        self.meta = json.loads((Path(ckpt) / "meta.json").read_text())

    @torch.no_grad()
    def chunk(self, joints, wrist, front, instruction) -> np.ndarray:
        """(50, 6) absolute owner-frame joint goals."""
        d = self.device
        img = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(d).permute(2, 0, 1)[None].float().div_(255)
        st = (np.asarray(joints, np.float32) - self.s["state_mean"]) / self.s["state_std"]
        toks, mask = self.tok([instruction], d)
        batch = {"observation.images.camera1": img(front), "observation.images.camera2": img(wrist),
                 "observation.state": torch.from_numpy(st)[None].to(d),
                 "observation.language.tokens": toks, "observation.language.attention_mask": mask}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.pol.predict_action_chunk(batch)
        a = a[0, :, :6].float().cpu().numpy()
        return a * self.s["action_std"] + self.s["action_mean"]
