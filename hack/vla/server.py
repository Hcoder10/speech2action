"""SmolVLA policy server (LeRobot 0.6.1 venv, needs a GPU lease). One client at a time; each request is
{joints, wrist, front, instr} -> (50, 6) absolute owner-frame joint goals.

  HF_HOME=runs/hf C:/Users/sarta/reachy-duplex/.venv-so101/Scripts/python.exe -m hack.vla.server --ckpt runs/vla_ckpt
"""
from __future__ import annotations

import argparse
import time
from multiprocessing.connection import Listener

import numpy as np

from .skill import ADDR, AUTH, _pack


def _unpack(t):
    b, shape, dt = t
    return np.frombuffer(b, dtype=dt).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/vla_ckpt")
    ap.add_argument("--port", type=int, default=ADDR[1])
    a = ap.parse_args()
    import os
    import torch
    from .policy import VLAPolicy
    vram = float(os.environ.get("LANDLORD_VRAM_GB", 0))
    if vram:
        torch.cuda.set_per_process_memory_fraction(min(vram / 15.92, 0.95))
    pol = VLAPolicy(a.ckpt)
    z = np.zeros((256, 256, 3), np.uint8)
    for _ in range(3):
        pol.chunk(np.zeros(6), z, z, "warm up")
    print(f"ready on {ADDR[0]}:{a.port} ckpt {a.ckpt} step {pol.meta.get('step')}", flush=True)
    with Listener((ADDR[0], a.port), authkey=AUTH) as L:
        while True:
            conn = L.accept()
            n, t_tot = 0, 0.0
            try:
                while True:
                    m = conn.recv()
                    t0 = time.perf_counter()
                    c = pol.chunk(_unpack(m["joints"]), _unpack(m["wrist"]), _unpack(m["front"]), m["instr"])
                    conn.send(_pack(c.astype(np.float32)))
                    n += 1; t_tot += time.perf_counter() - t0
            except (EOFError, ConnectionError, OSError):
                print(f"client gone after {n} requests, {1000 * t_tot / max(n, 1):.0f} ms/request", flush=True)


if __name__ == "__main__":
    main()
