"""PersonaPlex full-duplex client: laptop mic -> model, model voice -> laptop speaker, model words -> callback.

Wire protocol (moshi server, /api/chat websocket, binary frames, first byte = kind):
  0x00 handshake (server ready; start streaming)   0x01 Ogg/Opus audio, both directions (24 kHz mono)
  0x02 UTF-8 text piece of the model's own words   (anything else is ignored)
The model listens continuously, also while it talks and while the arm moves: that is the point.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from urllib.parse import quote

import numpy as np

MODEL_RATE = 24000
DEV_RATE = 48000
CHUNK = 480                    # 20 ms at 24 kHz


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x.astype(np.float32, copy=False)
    n = int(round(x.size * dst / src))
    return np.interp(np.linspace(0, 1, n, endpoint=False), np.linspace(0, 1, x.size, endpoint=False), x).astype(np.float32)


class Ring:
    """Thread-safe float32 FIFO with a cap (drops the oldest audio instead of building latency)."""

    def __init__(self, cap: int):
        self.buf = np.zeros(0, np.float32)
        self.cap = cap
        self.lock = threading.Lock()

    def push(self, x):
        with self.lock:
            self.buf = np.concatenate([self.buf, np.asarray(x, np.float32)])[-self.cap:]

    def pop(self, n, pad=True):
        with self.lock:
            out, self.buf = self.buf[:n], self.buf[n:]
        if pad and out.size < n:
            out = np.pad(out, (0, n - out.size))
        return out

    @property
    def depth(self):
        return self.buf.size


class PersonaPlex:
    def __init__(self, url: str, text_prompt: str, voice: str = "NATF2.pt", seed: int = 42,
                 on_text=None, on_level=None, on_state=None, in_device=None, out_device=None,
                 mic_gain: float = 1.0, duck: float = 0.35, volume: float = 1.0):
        sep = "&" if "?" in url else "?"
        self.url = f"{url}{sep}voice_prompt={quote(voice)}&text_prompt={quote(text_prompt)}&seed={seed}"
        self.on_text = on_text or (lambda s: None)
        self.on_level = on_level or (lambda who, lvl: None)
        self.on_state = on_state or (lambda s: None)
        self.in_device, self.out_device = in_device, out_device
        self.mic_gain, self.duck, self.volume = mic_gain, duck, volume
        self.mic = Ring(int(MODEL_RATE * 0.6))
        self.spk = Ring(int(MODEL_RATE * 1.5))
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.agent_speaking_until = 0.0
        self.muted = False
        self._streams = []

    # ------------------------------------------------------------------ audio devices
    def _start_audio(self):
        import sounddevice as sd

        info = sd.query_devices(self.in_device if self.in_device is not None else sd.default.device[0])
        in_rate = int(info.get("default_samplerate") or DEV_RATE)

        def mic_cb(indata, frames, t, status):
            x = indata[:, 0].astype(np.float32)
            g = 0.0 if self.muted else self.mic_gain
            if time.perf_counter() < self.agent_speaking_until:      # laptop speakers: don't let it hear itself
                g *= self.duck
            lvl = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
            self.on_level("user", lvl * (g > 0))
            self.mic.push(resample(x, in_rate, MODEL_RATE) * g)

        def spk_cb(outdata, frames, t, status):
            need = int(round(frames * MODEL_RATE / DEV_RATE))
            pcm = self.spk.pop(need)
            y = resample(pcm, MODEL_RATE, DEV_RATE)
            y = np.pad(y, (0, max(0, frames - y.size)))[:frames]
            outdata[:, 0] = np.clip(y * self.volume, -1, 1)

        i = sd.InputStream(samplerate=in_rate, channels=1, dtype="float32", blocksize=in_rate // 50,
                           device=self.in_device, callback=mic_cb)
        o = sd.OutputStream(samplerate=DEV_RATE, channels=1, dtype="float32", blocksize=960,
                            device=self.out_device, callback=spk_cb)
        i.start(); o.start()
        self._streams = [i, o]
        return info["name"], sd.query_devices(o.device)["name"]

    # ------------------------------------------------------------------ network
    async def run(self):
        import sphn
        import websockets

        enc, dec = sphn.OpusStreamWriter(MODEL_RATE), sphn.OpusStreamReader(MODEL_RATE)
        self.on_state("connecting")
        async with websockets.connect(self.url, max_size=None, open_timeout=30, ping_interval=None) as ws:
            self.on_state("waiting for model")
            mic_name, spk_name = self._start_audio()

            async def uplink():
                await self.ready.wait()
                self.mic.pop(self.mic.depth)
                nxt = time.perf_counter()
                while not self.stop.is_set():
                    nxt += 0.02
                    enc.append_pcm(self.mic.pop(CHUNK))
                    data = enc.read_bytes()
                    if data:
                        with contextlib.suppress(Exception):
                            await ws.send(b"\x01" + bytes(data))
                    await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

            async def downlink():
                async for msg in ws:
                    if not isinstance(msg, (bytes, bytearray)) or not msg:
                        continue
                    kind, payload = msg[0], bytes(msg[1:])
                    if kind == 0x00:
                        self.ready.set()
                        self.on_state(f"live (mic: {mic_name} | speaker: {spk_name})")
                    elif kind == 0x01:
                        dec.append_bytes(payload)
                        pcm = dec.read_pcm()
                        if pcm is not None and len(pcm):
                            pcm = np.asarray(pcm, np.float32).reshape(-1)
                            self.spk.push(pcm)
                            lvl = float(np.sqrt(np.mean(pcm ** 2)))
                            self.on_level("agent", lvl)
                            if lvl > 0.01:
                                self.agent_speaking_until = time.perf_counter() + self.spk.depth / MODEL_RATE + 0.25
                    elif kind == 0x02:
                        self.on_text(payload.decode("utf-8", errors="replace"))

            up = asyncio.create_task(uplink())
            try:
                await downlink()
            finally:
                self.stop.set()
                up.cancel()
                for s in self._streams:
                    with contextlib.suppress(Exception):
                        s.stop(); s.close()
                self.on_state("disconnected")
