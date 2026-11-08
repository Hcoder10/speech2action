"""Headless check of the whole stack without voice: scripted agent sentences -> parser -> skills -> sim.

  python -m hack.sim_script [--model ik|rl] [--video out.mp4]
"""
import argparse
import time

import numpy as np

from .intent import IntentParser
from .scene import DemoEnv
from .skills import IKController, Kinematics, Skills

SCRIPT = [  # (seconds after start, what the agent says)
    (0.5, "Sure! Grabbing the red cube."),
    (7.0, "Putting it in the bowl."),
    (13.0, "Now I'll pick up the blue one."),
    (15.0, "Oh, not the blue, the green one."),
    (21.0, "Lifting it a bit higher."),
    (23.0, "And I'll put it on the left."),
    (29.0, "Let me point at the blue cube."),
    (33.0, "Hello! *waves*"),
    (38.0, "Going back home."),
]


def make_controller(kind, kin):
    if kind == "rl":
        from .action_model.model import ActionModel
        return ActionModel()
    return IKController(kin)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ik"); ap.add_argument("--video"); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    env = DemoEnv(seed=0); obs, _ = env.reset(seed=a.seed)
    kin = Kinematics(); sk = Skills(make_controller(a.model, kin), kin); parser = IntentParser()
    frames, script, t0 = [], list(SCRIPT), time.time()
    for k in range(int(42 * 30)):
        t = k / 30
        while script and script[0][0] <= t:
            _, line = script.pop(0)
            for it in parser.feed_text(line):
                print(f"{t:5.1f}s  '{line}' -> {it}")
                sk.handle(it, parser.last_obj)
        obs, *_ = env.step(sk.tick(obs, env.objects()))
        for e in sk.events:
            print(f"{t:5.1f}s  event: {e}")
        sk.events.clear()
        if a.video and k % 2 == 0:
            frames.append(env.render_view(w=640, h=480))
    o = env.objects()
    in_bowl = lambda n: np.linalg.norm(o[n]["pos"][:2] - o["bowl"]["pos"][:2]) < 0.06
    print(f"wall {time.time() - t0:.1f}s for 42s sim | red in bowl: {in_bowl('red')} | green at {o['green']['pos'].round(3)} "
          f"| final skill {sk.status.skill} {sk.status.phase}")
    if a.video:
        import cv2
        w = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
        for f in frames: w.write(f[..., ::-1])
        w.release(); print("wrote", a.video)


if __name__ == "__main__":
    main()
