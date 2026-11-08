"""Demo-video post-production: remove Pip's echo from the mic track, re-mix, and burn in captions.

  python -m hack.postprod runs/live_session9

Inputs written by the app: <name>.mp4 (video), <name>.wav (mix), <name>_mic.wav (raw mic), <name>_speaker.wav (exactly
what Pip played), <name>.jsonl (events, incl. Pip's text pieces). Output: <name>_final.mp4.
  * echo removal: the speaker track is the reference. Both tracks are located in the mix by cross-correlation; a
    per-frequency echo path |H(f)| is fitted where the user is quiet, and the predicted echo is subtracted from the
    mic spectrum (with a little over-subtraction and a floor), keeping the mic phase.
  * captions: Pip's own words come from its text stream (exact); the user's come from Whisper on the cleaned mic.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 24000


def rd(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def wr(p, x):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32000).astype(np.int16).tobytes())


def locate(sub, mix, max_lag_s=8.0):
    """Offset (samples) of `sub` inside `mix`, by FFT cross-correlation of envelopes."""
    n = min(len(mix), len(sub) + int(max_lag_s * SR))
    a, b = np.abs(mix[:n]), np.abs(sub[:n])
    f = np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n))
    c = np.fft.irfft(f)[: int(max_lag_s * SR)]
    return int(np.argmax(c))


def stft(x, n=1024, h=256):
    w = np.hanning(n).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (n, n)), n)[::h] * w
    return np.fft.rfft(frames, axis=1)


def istft(X, length, n=1024, h=256):
    w = np.hanning(n).astype(np.float32)
    frames = np.fft.irfft(X, n=n, axis=1) * w
    out = np.zeros(len(frames) * h + n, np.float32); norm = np.zeros_like(out)
    for i, fr in enumerate(frames):
        out[i * h:i * h + n] += fr; norm[i * h:i * h + n] += w ** 2
    out /= np.maximum(norm, 1e-6)
    return out[n:n + length]


def deecho(mic, ref):
    """Spectral echo subtraction with the speaker track as reference (same length, already aligned)."""
    M, R = stft(mic), stft(ref)
    best = (None, -1)
    for d in range(0, 24):                        # acoustic + device delay, in 256-sample hops (~0-250 ms)
        Rd = np.roll(np.abs(R), d, axis=0); Rd[:d] = 0
        num = np.sum(np.abs(M) * Rd); den = np.sqrt(np.sum(np.abs(M) ** 2) * np.sum(Rd ** 2)) + 1e-9
        if num / den > best[1]:
            best = (d, num / den)
    d = best[0]
    Rd = np.roll(np.abs(R), d, axis=0); Rd[:d] = 0
    ref_on = Rd.sum(1) > np.percentile(Rd.sum(1), 60)       # frames where Pip is clearly playing
    H = np.sum(np.abs(M[ref_on]) * Rd[ref_on], 0) / (np.sum(Rd[ref_on] ** 2, 0) + 1e-9)
    H = np.clip(H, 0, None)
    echo = 1.5 * H * Rd
    mag = np.maximum(np.abs(M) - echo, 0.08 * np.abs(M))
    gain = mag / (np.abs(M) + 1e-9)
    # frames dominated by echo (user quiet): suppress hard
    ratio = np.sum(echo, 1) / (np.sum(np.abs(M), 1) + 1e-9)
    gain[ratio > 0.6] *= 0.15
    return istft(M * gain, len(mic)), d * 256 / SR


def srt_time(t):
    t = max(0.0, t); h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def pip_lines(jsonl, t_video0):
    """Group Pip's text pieces into caption lines at pauses / sentence ends."""
    lines, cur, t_start, t_last = [], "", None, None
    for line in open(jsonl, encoding="utf-8"):
        t, j = line.split(" ", 1); t = float(t); j = json.loads(j)
        if j["type"] != "agent_text" or j.get("typed"):
            continue
        piece = j["text"]
        if t_last is not None and (t - t_last > 0.9 or (cur.rstrip().endswith((".", "!", "?")) and len(cur) > 25)):
            lines.append((t_start, t_last + 0.6, cur.strip())); cur, t_start = "", None
        if t_start is None:
            t_start = t
        cur += piece; t_last = t
    if cur.strip():
        lines.append((t_start, t_last + 0.6, cur.strip()))
    return [(a - t_video0, b - t_video0, f"Pip: {txt}") for a, b, txt in lines if txt]


def main(base):
    base = Path(base)
    mix, mic, spk = rd(f"{base}.wav"), rd(f"{base}_mic.wav"), rd(f"{base}_speaker.wav")
    o_spk = locate(spk, mix); o_mic = locate(mic, mix)
    L = len(mix)
    S = np.zeros(L, np.float32); S[o_spk:o_spk + len(spk)] = spk[:max(0, L - o_spk)]
    U = np.zeros(L, np.float32); U[o_mic:o_mic + len(mic)] = mic[:max(0, L - o_mic)]
    clean, delay = deecho(U, S)
    print(f"speaker at {o_spk / SR:.2f}s, mic at {o_mic / SR:.2f}s in the mix; echo delay {delay * 1000:.0f} ms")
    wr(f"{base}_user_clean.wav", clean)
    out = 0.85 * S + 0.9 * clean
    out = np.tanh(out * 1.2) / np.tanh(1.2)
    wr(f"{base}_clean.wav", out)

    # captions
    first_t = float(open(f"{base}.jsonl", encoding="utf-8").readline().split(" ", 1)[0])
    caps = pip_lines(f"{base}.jsonl", first_t)
    try:
        if os.environ.get("NO_WHISPER"):
            raise RuntimeError("skipped (NO_WHISPER)")
        os.environ.setdefault("HF_HOME", str(Path("runs/whisper").resolve()))
        from faster_whisper import WhisperModel
        m = WhisperModel("small.en", device="cpu", compute_type="int8")
        x16 = np.interp(np.arange(0, len(clean), 1.5), np.arange(len(clean)), clean).astype(np.float32)   # 24k -> 16k
        segs, _ = m.transcribe(x16, vad_filter=True, beam_size=1)
        for s in segs:
            txt = s.text.strip()
            if txt:
                caps.append((s.start, s.end, f"You: {txt}"))
    except Exception as e:
        print("whisper failed:", e)
    caps.sort()
    srt = Path(f"{base}.srt")
    with open(srt, "w", encoding="utf-8") as f:
        for i, (a, b, txt) in enumerate(caps, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n")
    print(f"{len(caps)} caption lines -> {srt}")
    style = "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=24"
    sub = srt.as_posix()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", f"{base}.mp4", "-i", f"{base}_clean.wav",
           "-vf", f"subtitles='{sub}':force_style='{style}'", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-c:a", "aac", "-b:a", "160k", "-shortest", f"{base}_final.mp4"]
    subprocess.run(cmd, check=True)
    print("wrote", f"{base}_final.mp4")


if __name__ == "__main__":
    main(sys.argv[1])
