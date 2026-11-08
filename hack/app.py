"""The demo: talk to PersonaPlex, the SO-101 does what it says it is doing.

  python -m hack.app                          # sim + PersonaPlex at ws://127.0.0.1:8998/api/chat, UI on :8080
  python -m hack.app --no-voice               # sim only; type what the agent would say in the UI
  python -m hack.app --model rl               # learned action model instead of the IK fallback

Threads: the 30 Hz control loop (sim, skills, action model, rendering) runs in its own thread on a real-time
clock; asyncio runs the PersonaPlex client and the web server. They meet through a queue of intents and a slot
holding the latest JPEG frame.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from .intent import IntentParser
from .scene import COLORS, DemoEnv
from .skills import IKController, Kinematics, Skills

WEB = Path(__file__).resolve().parent / "web"


def persona(objects: dict) -> str:
    def where(p):
        side = "left" if p[1] < -0.05 else ("right" if p[1] > 0.05 else "middle")
        return f"on the {side}" if side != "middle" else "in the middle"
    things = ", ".join(f"a {c} cube {where(objects[c]['pos'])}" for c in COLORS)
    return (
        "You are Pip, a cheerful little robot arm sitting on a wooden table. You can see " + things +
        f", and a white bowl {where(objects['bowl']['pos'])}. You can pick up a cube, put it in the bowl or somewhere "
        "on the table or on top of another cube, point at things, move a little left, right, up or down, open or "
        "close your gripper, wave, nod, dance, go back home, and stop. You can also do bigger jobs: clean up the table "
        "(everything into the bowl), build a tower out of the cubes, or line them up in a row. When the person asks you to do something, "
        "say out loud what you are doing in plain words, like 'Okay, grabbing the red cube!' or 'Putting it in "
        "the bowl.' If they change their mind, say the new thing, like 'Oh, the blue one then.' If they say stop, "
        "say 'Stopping.' Keep every reply short, warm and a little playful."
    )


class Controller(threading.Thread):
    """The 30 Hz loop. Owns the sim, the GL context, the skills and the action model."""

    def __init__(self, args, emit):
        super().__init__(daemon=True)
        self.args, self.emit = args, emit
        self.intents: queue.Queue = queue.Queue()
        self.cmds: queue.Queue = queue.Queue()
        self.frame: bytes | None = None
        self.frame_id = 0
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.running = True
        self.objects = {}

    def make_ctrl(self, kin):
        if self.args.model == "rl":
            from .action_model.model import ActionModel
            return ActionModel()
        return IKController(kin)

    def run(self):
        a = self.args
        env = DemoEnv(seed=a.seed)
        obs, _ = env.reset(seed=a.seed)
        kin = Kinematics()
        sk = Skills(self.make_ctrl(kin), kin)
        self.objects = env.objects()
        arm = None
        if a.real:                       # mirror the sim's joint goals onto the physical SO-101
            from so101_sim2real.hw import Arm
            arm = Arm(a.port_name, allow_motion=True, stop_file=Path("runs") / "STOP")
            start = arm.read()
            print("real arm connected at", np.round(start, 1).tolist(), flush=True)
            arm.move_to(np.asarray(obs["prev_goal"], float), speed_deg_s=15.0)
        self.ready.set()
        period, tick = 1 / 30, time.perf_counter()
        writer = None
        last_status = 0.0
        while self.running:
            while not self.cmds.empty():
                c = self.cmds.get()
                if c == "reset":
                    obs, _ = env.reset(seed=int(time.time()) % 100000)
                    sk = Skills(self.make_ctrl(kin), kin)
                    self.objects = env.objects()
                    self.emit({"type": "scene", "persona": persona(self.objects)})
            while not self.intents.empty():
                it, last_obj = self.intents.get()
                sk.handle(it, last_obj)
                self.emit({"type": "intent", "kind": it.kind, "label": str(it), "text": it.text,
                           "latency_ms": round((time.perf_counter() - it.t) * 1000, 1)})
            obs, *_ = env.step(sk.tick(obs, env.objects()))
            if arm is not None:
                if arm.stop_requested():
                    arm.hold(); arm = None
                    self.emit({"type": "event", "text": "real arm: STOP file, holding"})
                else:
                    meas = arm.read()
                    goal = np.clip(obs["prev_goal"], meas - 10, meas + 10)     # LeRobot-style relative clamp
                    arm.write(goal)
            for e in sk.events:
                self.emit({"type": "event", "text": e})
            sk.events.clear()
            img = env.render_view("demo", a.width, a.height)
            if a.record:
                if writer is None:
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (a.width, a.height))
                    self.record_t0 = time.perf_counter()
                writer.write(img[..., ::-1])
            ok, jpg = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                self.frame, self.frame_id = jpg.tobytes(), self.frame_id + 1
            now = time.perf_counter()
            if now - last_status > 0.1:
                st = sk.status
                self.emit({"type": "status", "skill": st.skill, "phase": st.phase, "holding": st.holding,
                           "stopped": st.stopped, "tip": st.tip, "grip": round(float(obs["joints"][5]))})
                last_status = now
            tick += period
            delay = tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                tick = time.perf_counter()        # fell behind: don't try to catch up in a burst
        if writer is not None:
            writer.release()
        self.stopped.set()


class App:
    def __init__(self, args):
        self.args = args
        self.clients: set[web.WebSocketResponse] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.parser = IntentParser()
        self.last_text_t = 0.0
        self.ctrl = Controller(args, self.emit_threadsafe)
        self.voice = None
        self.probe = None
        self.last_user_voice = -1e9
        if args.hidden:
            from .hidden.probe import HiddenProbe
            self.probe = HiddenProbe()

    # ------------------------------------------------------------------ events to the UI
    def emit_threadsafe(self, msg: dict):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._broadcast, json.dumps(msg))

    def emit(self, msg: dict):
        self._broadcast(json.dumps(msg))

    def _broadcast(self, s: str):
        if self.args.log:
            with open(self.args.log, "a", encoding="utf-8") as f:
                f.write(f"{time.time():.3f} {s}" + "\n")
        for ws in list(self.clients):
            if not ws.closed:
                asyncio.ensure_future(ws.send_str(s))

    # ------------------------------------------------------------------ agent words -> intents
    def on_agent_text(self, piece: str):
        self.last_text_t = time.perf_counter()
        self.emit({"type": "agent_text", "text": piece})
        for it in self.parser.feed(piece):
            self.ctrl.intents.put((it, self.parser.last_obj))

    def on_typed(self, text: str):
        self.emit({"type": "agent_text", "text": " " + text + " ", "typed": True})
        for it in self.parser.feed_text(text):
            self.ctrl.intents.put((it, self.parser.last_obj))

    async def flusher(self):
        """A pause in the model's speech ends the utterance: resolve any phrase still waiting for words."""
        while True:
            await asyncio.sleep(0.1)
            if self.parser.partial or len(self.parser.words) > self.parser.consumed:
                if time.perf_counter() - self.last_text_t > 0.6:
                    for it in self.parser.flush():
                        self.ctrl.intents.put((it, self.parser.last_obj))

    # ------------------------------------------------------------------ web
    async def index(self, request):
        return web.FileResponse(WEB / "index.html")

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(json.dumps({"type": "scene", "persona": persona(self.ctrl.objects),
                                      "voice": self.args.no_voice is False, "model": self.args.model}))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    if m.get("type") == "say":
                        self.on_typed(m["text"])
                    elif m.get("type") == "reset":
                        self.ctrl.cmds.put("reset")
                    elif m.get("type") == "mute" and self.voice is not None:
                        self.voice.muted = bool(m.get("on"))
        finally:
            self.clients.discard(ws)
        return ws

    async def frame_pump(self):
        last = -1
        while True:
            await asyncio.sleep(1 / 30)
            if self.ctrl.frame_id != last and self.ctrl.frame is not None:
                last = self.ctrl.frame_id
                for ws in list(self.clients):
                    if not ws.closed:
                        try:
                            await ws.send_bytes(self.ctrl.frame)
                        except Exception:
                            pass

    def on_level(self, who, lvl):
        if who == "user" and lvl > 0.015:
            self.last_user_voice = time.perf_counter()

    def on_hidden(self, h, k):
        """Every 80 ms: the probe reads PersonaPlex's hidden state; confident, sustained predictions act at once."""
        # gate: the probe may act only in the ~1.5 s right after the user spoke (mic energy), which is where it was
        # trained to look; on held-out sessions that removed every spurious fire during the agent's own speech
        its = self.probe.push(h)
        if time.perf_counter() - self.last_user_voice > 1.5:
            its = []
        for it in its:
            if it.obj:
                self.parser.last_obj = it.obj
            if it.kind in ("pick", "point", "place"):
                self.parser.last_kind = it.kind
            self.ctrl.intents.put((it, self.parser.last_obj))
        if k % 2 == 0:
            from .hidden.probe import CLASSES
            p = self.probe.last_probs
            top = np.argsort(p)[::-1][:4]
            self.emit({"type": "hidden", "top": [[CLASSES[i], round(float(p[i]), 3)] for i in top],
                       "threshold": self.probe.th})

    async def run_voice(self):
        from .voice import PersonaPlex

        while True:
            self.voice = PersonaPlex(
                self.args.pp_url, persona(self.ctrl.objects), voice=self.args.voice, on_text=self.on_agent_text,
                on_level=self.on_level, in_device=self.args.in_device, out_device=self.args.out_device,
                on_state=lambda s: self.emit({"type": "voice", "state": s}), mic_wav=self.args.mic_wav,
                record_audio=(self.args.record[:-4] + ".wav") if self.args.record else None,
                record_t0=lambda: getattr(self.ctrl, "record_t0", None),
                speaker=not self.args.no_speaker,
                on_hidden=self.on_hidden if self.probe is not None else None)
            try:
                await self.voice.run()
            except Exception as e:
                self.emit({"type": "voice", "state": f"error: {e!r}"[:200]})
            await asyncio.sleep(3)

    async def main(self):
        self.loop = asyncio.get_running_loop()
        self.ctrl.start()
        await asyncio.get_running_loop().run_in_executor(None, self.ctrl.ready.wait)
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_static("/static", WEB)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", self.args.port).start()
        print(f"UI: http://localhost:{self.args.port}", flush=True)
        tasks = [asyncio.create_task(self.frame_pump()), asyncio.create_task(self.flusher())]
        if not self.args.no_voice:
            tasks.append(asyncio.create_task(self.run_voice()))
        try:
            if self.args.duration:
                await asyncio.sleep(self.args.duration)
            else:
                await asyncio.gather(*tasks)
        finally:                                   # clean shutdown: close the video, save the agent audio
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.ctrl.running = False
            await asyncio.get_running_loop().run_in_executor(None, self.ctrl.stopped.wait, 5)
            await runner.cleanup()
            if self.args.record:
                mux(self.args.record)


def mux(mp4: str):
    """demo.mp4 + demo.wav -> demo_av.mp4 (h264 + aac) if ffmpeg is around."""
    import shutil
    import subprocess
    wav = mp4[:-4] + ".wav"
    ff = shutil.which("ffmpeg")
    if ff and Path(wav).exists():
        out = mp4[:-4] + "_av.mp4"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4, "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-c:a", "aac", "-shortest", out], check=False)
        print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp-url", default="ws://127.0.0.1:8998/api/chat")
    ap.add_argument("--voice", default="NATF2.pt")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--model", choices=("ik", "rl"), default="ik")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--in-device", type=int)
    ap.add_argument("--out-device", type=int)
    ap.add_argument("--mic-wav", help="scripted user audio instead of the microphone")
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--hidden", action="store_true", help="act on the hidden-state probe as well as the words")
    ap.add_argument("--duration", type=float, help="run this many seconds, then shut down cleanly")
    ap.add_argument("--record", help="write the demo view to this .mp4 (agent audio goes to <name>.wav)")
    ap.add_argument("--log", help="append every UI event as JSON lines here")
    ap.add_argument("--real", action="store_true", help="mirror the sim onto the SO-101 (owner at the arm)")
    ap.add_argument("--port-name", dest="port_name", default="COM3")
    a = ap.parse_args()
    asyncio.run(App(a).main())


if __name__ == "__main__":
    main()
