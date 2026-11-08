"""Record PersonaPlex hidden states for scripted user sessions (probe training data).

Each episode: ~7 TTS user utterances with gaps, streamed in real time into a live server as if from a mic.
Saved per episode (runs/hidden/ep_XXXX.npz): hidden (T, 4096) float16, one per 80 ms frame; the frame index of
every agent text piece; the user schedule (label, start/end in frames). Frame k of the hidden stream covers uplink
audio frame k counted from the handshake (the server processes audio strictly in order), so labels align exactly
even when the network or the GPU lags.

  python -m hack.hidden.collect --servers ws://127.0.0.1:8998/api/chat ws://127.0.0.1:8999/api/chat --episodes 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import wave
from pathlib import Path

import numpy as np

from ..app import persona
from ..voice import MODEL_RATE, PersonaPlex
from .phrases import COLORS, PHRASES

TTS = Path("runs/tts")
OUT = Path("runs/hidden")
FRAME = 1920                      # 80 ms at 24 kHz


def load(i, v):
    with wave.open(str(TTS / f"p{i}_v{v}.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def make_episode(rng: random.Random):
    """A plausible session: picks, places, corrections, stops, nudges, chit-chat, multi-step jobs."""
    by = {}
    for i, (lab, _) in enumerate(PHRASES):
        by.setdefault(lab, []).append(i)
    plan = []
    for _ in range(7):
        r = rng.random()
        if r < 0.35:
            lab = f"pick_{rng.choice(COLORS)}"
        elif r < 0.50:
            lab = "place_bowl"
        elif r < 0.62:
            lab = "stop"
        elif r < 0.72:
            lab = rng.choice(["nudge_left", "nudge_right", "nudge_up"])
        elif r < 0.80:
            lab = rng.choice(["wave", "home"])
        elif r < 0.88:
            lab = rng.choice(["cleanup", "tower"])
        else:
            lab = "none"
        plan.append(lab)
    v = rng.randrange(2)
    audio, sched, pos = [np.zeros(int(MODEL_RATE * rng.uniform(1.5, 3.0)), np.float32)], [], 0
    pos = audio[0].size
    for lab in plan:
        i = rng.choice(by[lab])
        x = load(i, v if rng.random() < 0.8 else 1 - v) * rng.uniform(0.6, 1.0)
        sched.append({"label": lab, "text": PHRASES[i][1], "start": pos / FRAME, "end": (pos + x.size) / FRAME})
        gap = np.zeros(int(MODEL_RATE * (rng.uniform(1.0, 2.5) if lab == "stop" else rng.uniform(3.5, 6.5))), np.float32)
        audio += [x, gap]
        pos += x.size + gap.size
    return np.concatenate(audio), sched


def fake_layout(rng):
    sides = {"left": np.array([0.25, -0.12, 0.015]), "middle": np.array([0.26, 0.0, 0.015]), "right": np.array([0.25, 0.12, 0.015])}
    order = rng.sample(list(sides), 3)
    objs = {c: {"pos": sides[s]} for c, s in zip(COLORS, order)}
    objs["bowl"] = {"pos": np.array([0.22, rng.choice([-0.16, 0.16]), 0.0])}
    return objs


async def run_episode(url, idx, seed):
    rng = random.Random(seed)
    audio, sched = make_episode(rng)
    wav_path = OUT / f"ep_{idx:04d}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(MODEL_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    hidden, texts = [], []
    total_frames = audio.size // FRAME + int(3.0 * 12.5)
    done = asyncio.Event()

    def on_hidden(h, k):
        hidden.append(h.copy())
        if len(hidden) >= total_frames:
            done.set()

    pp = PersonaPlex(url, persona(fake_layout(rng)), voice="NATF2.pt", seed=seed, mic_wav=str(wav_path), speaker=False,
                     on_hidden=on_hidden, on_text=lambda s: texts.append((len(hidden) - 1, s)))
    task = asyncio.create_task(pp.run())
    try:
        await asyncio.wait_for(done.wait(), timeout=audio.size / MODEL_RATE + 60)
    except asyncio.TimeoutError:
        pass
    pp.stop.set()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    if len(hidden) < total_frames * 0.8:
        print(f"ep {idx}: only {len(hidden)}/{total_frames} frames, dropped", flush=True)
        return
    np.savez_compressed(OUT / f"ep_{idx:04d}.npz", hidden=np.stack(hidden), text_frames=np.array([f for f, _ in texts]),
                        text=json.dumps([t for _, t in texts]), schedule=json.dumps(sched), seed=seed)
    print(f"ep {idx} via {url[-14:]}: {len(hidden)} frames, agent said: {''.join(t for _, t in texts)[:120]}", flush=True)


async def worker(url, queue):
    while True:
        try:
            idx, seed = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            await run_episode(url, idx, seed)
        except Exception as e:
            print(f"ep {idx}: {e!r}", flush=True)
        await asyncio.sleep(1.0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", nargs="+", default=["ws://127.0.0.1:8999/api/chat"])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    q: asyncio.Queue = asyncio.Queue()
    for i in range(a.start, a.start + a.episodes):
        if not (OUT / f"ep_{i:04d}.npz").exists():
            q.put_nowait((i, 1000 + i))
    await asyncio.gather(*(worker(u, q) for u in a.servers))


if __name__ == "__main__":
    asyncio.run(main())
