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
                 mic_gain: float = 1.0, duck: float = 0.7, volume: float = 1.0, mic_wav: str | None = None,
                 speaker: bool = True, record_audio: str | None = None, record_t0=None, on_hidden=None,
                 on_probs=None):
        sep = "&" if "?" in url else "?"
        self.url = f"{url}{sep}voice_prompt={quote(voice)}&text_prompt={quote(text_prompt)}&seed={seed}"
        self.on_hidden = on_hidden                # float16[4096] per 80 ms frame (needs the hidden-stream server patch)
        if on_hidden is not None:
            self.url += "&hidden=1"
        self.on_probs = on_probs                  # 13 class probabilities per frame, probe run ON the server
        if on_probs is not None:
            self.url += "&probe=1"
        self.frames = 0
        self.on_text = on_text or (lambda s: None)
        self.on_level = on_level or (lambda who, lvl: None)
        self.on_state = on_state or (lambda s: None)
        self.in_device, self.out_device = in_device, out_device
        self.mic_gain, self.duck, self.volume = mic_gain, duck, volume
        self.barge = 99.0                         # echo-free: mic fully muted while Pip speaks (no barge-in)
        self.mic = Ring(int(MODEL_RATE * 0.6))
        self.spk = Ring(int(MODEL_RATE * 0.8))      # never let Pip fall more than 0.8 s behind
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.agent_speaking_until = 0.0
        self.muted = False
        self._streams = []
        self.speaker = speaker
        self.record_audio, self.record_t0 = record_audio, record_t0 or (lambda: None)
        self.rec_chunks: list = []                # (wall time, pcm) of the agent's voice, for the demo video
        self.mic_rec: list = []                   # contiguous 24 kHz mic stream, for the demo video
        self.out_rec: list = []                   # contiguous 24 kHz speaker stream (what the user heard)
        self.mic_t0 = self.out_t0 = None
        self.underruns = 0
        self.wav = None
        if mic_wav:                               # scripted user turns instead of a microphone (tests, rehearsals)
            import wave
            with wave.open(mic_wav) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
                self.wav = resample(x, w.getframerate(), MODEL_RATE)
            self.wav_pos = 0

    # ------------------------------------------------------------------ audio devices
    def _start_audio(self):
        import sounddevice as sd

        info = sd.query_devices(self.in_device if self.in_device is not None else sd.default.device[0])
        in_rate = int(info.get("default_samplerate") or DEV_RATE)

        def mic_cb(indata, frames, t, status):
            x = indata[:, 0].astype(np.float32)
            g = 0.0 if self.muted else self.mic_gain
            lvl0 = float(np.sqrt(np.mean(x ** 2)))
            if time.perf_counter() < self.agent_speaking_until:      # laptop speakers: don't let it hear itself
                # echo gate: while Pip talks, only a loud barge-in (the user right at the laptop) gets through
                if lvl0 >= self.barge:
                    self._barge_until = time.perf_counter() + 0.6
                g = g if time.perf_counter() < getattr(self, "_barge_until", 0.0) else 0.0
            lvl = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
            speaking = time.perf_counter() < self.agent_speaking_until
            # while Pip talks through laptop speakers its echo reaches the mic: only loud speech counts as the user
            self.on_level("user", 0.0 if self.muted or (speaking and lvl < 0.05) else lvl)
            y = resample(x, in_rate, MODEL_RATE)
            if self.record_audio:                       # contiguous stream; only the first timestamp is used
                if self.mic_t0 is None:
                    self.mic_t0 = time.perf_counter() - frames / in_rate
                self.mic_rec.append(np.tanh(y * self.mic_gain * 1.5) / 1.5)
            self.mic.push(np.tanh(y * g * 1.5) / 1.5)          # soft-limit what the model hears

        self._playing = False

        def spk_cb(outdata, frames, t, status):
            need = int(round(frames * MODEL_RATE / DEV_RATE))
            # jitter buffer: start (and restart after a dry spell) only once 120 ms is queued, never play half chunks
            if not self._playing and self.spk.depth >= int(0.12 * MODEL_RATE):
                self._playing = True
            if self._playing and self.spk.depth >= need:
                pcm = self.spk.pop(need)
            else:
                if self._playing and self.spk.depth < need:
                    self.underruns += 1
                self._playing = False
                pcm = np.zeros(need, np.float32)
            if hasattr(self, "_last_out"):                  # resample across block edges without a seam
                y_in = np.r_[self._last_out[-1:], pcm]
            self._last_out = pcm
            if self.record_audio:                       # exactly what came out of the speaker, contiguous
                if self.out_t0 is None:
                    self.out_t0 = time.perf_counter()
                self.out_rec.append(pcm.copy())
            y = resample(pcm, MODEL_RATE, DEV_RATE)
            y = np.pad(y, (0, max(0, frames - y.size)))[:frames]
            outdata[:, 0] = np.clip(y * self.volume, -1, 1)

        names = ["scripted wav", "none"]
        if self.wav is None:
            i = sd.InputStream(samplerate=in_rate, channels=1, dtype="float32", blocksize=in_rate // 50,
                               device=self.in_device, callback=mic_cb)
            i.start(); self._streams.append(i); names[0] = info["name"]
        if self.speaker:
            o = sd.OutputStream(samplerate=DEV_RATE, channels=1, dtype="float32", blocksize=960,
                                device=self.out_device, callback=spk_cb)
            o.start(); self._streams.append(o); names[1] = sd.query_devices(o.device)["name"]
        return tuple(names)

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
                    if self.wav is not None:
                        piece = self.wav[self.wav_pos:self.wav_pos + CHUNK]; self.wav_pos += CHUNK
                        self.mic.push(np.pad(piece, (0, CHUNK - piece.size)))
                        self.on_level("user", float(np.sqrt(np.mean(piece ** 2))) if piece.size else 0.0)
                    if not self.speaker:
                        self.spk.pop(CHUNK)                              # nobody listens: keep the queue drained
                    data = enc.append_pcm(self.mic.pop(CHUNK))          # sphn >= 0.2 returns the bytes
                    if data is None and hasattr(enc, "read_bytes"):        # sphn 0.1.x
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
                        self.ready_t = time.perf_counter()
                        self.on_state(f"live (mic: {mic_name} | speaker: {spk_name})")
                    elif kind == 0x01:
                        pcm = dec.append_bytes(payload)                     # sphn >= 0.2 returns the pcm
                        if pcm is None and hasattr(dec, "read_pcm"):
                            pcm = dec.read_pcm()
                        if pcm is not None and len(pcm):
                            pcm = np.asarray(pcm, np.float32).reshape(-1)
                            self.spk.push(pcm)
                            if self.record_audio:
                                self.rec_chunks.append((time.perf_counter(), pcm))
                            lvl = float(np.sqrt(np.mean(pcm ** 2)))
                            self.on_level("agent", lvl)
                            if lvl > 0.01:
                                self.agent_speaking_until = time.perf_counter() + self.spk.depth / MODEL_RATE + 0.5
                    elif kind == 0x05:
                        self.frames += 1
                        if self.on_hidden is not None:
                            self.on_hidden(np.frombuffer(payload, np.float16), self.frames)
                    elif kind == 0x06:
                        self.frames += 1
                        if self.on_probs is not None:
                            self.on_probs(np.frombuffer(payload, np.float32), self.frames)
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
                self._save_audio()

    def _save_audio(self):
        t0 = self.record_t0()
        if not self.record_audio or t0 is None or getattr(self, "_saved", False):
            return
        import wave
        tracks = []                                         # (start offset s, samples, gain)
        if self.out_rec and self.out_t0 is not None:
            tracks.append((self.out_t0 - t0, np.concatenate(self.out_rec), 0.8))
        elif self.rec_chunks:                               # no speaker (scripted runs): lay network chunks end to end
            first = self.rec_chunks[0][0]
            tracks.append((first - t0, np.concatenate([p for _, p in self.rec_chunks]), 0.8))
        if self.mic_rec and self.mic_t0 is not None:
            tracks.append((self.mic_t0 - t0, np.concatenate(self.mic_rec), 0.7))
        if not tracks:
            return
        end = max(off + x.size / MODEL_RATE for off, x, _ in tracks) + 1
        out = np.zeros(int(end * MODEL_RATE), np.float32)
        for off, x, g in tracks:
            a = int(off * MODEL_RATE)
            if a < 0:
                x, a = x[-a:], 0
            seg = x[:max(0, out.size - a)]
            out[a:a + seg.size] += g * seg
        if self.wav is not None:                            # mix the scripted user voice in too
            u0 = int((self.ready_t - t0) * MODEL_RATE)
            u = self.wav[max(0, -u0):]
            out[max(0, u0):max(0, u0) + u.size] += 0.8 * u[:max(0, out.size - max(0, u0))]
        for name, x in (("speaker", np.concatenate(self.out_rec) if self.out_rec else None),
                        ("mic", np.concatenate(self.mic_rec) if self.mic_rec else None)):
            if x is not None:                               # raw tracks, for debugging the mix
                with wave.open(self.record_audio[:-4] + f"_{name}.wav", "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(MODEL_RATE)
                    w.writeframes((np.clip(x, -1, 1) * 32000).astype(np.int16).tobytes())
        print(f"speaker underruns: {self.underruns}", flush=True)
        out = np.tanh(out * 1.2) / np.tanh(1.2)            # soft limiter instead of hard clipping
        with wave.open(self.record_audio, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(MODEL_RATE)
            w.writeframes((out * 32000).astype(np.int16).tobytes())
        self._saved = True
