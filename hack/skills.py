"""Skill executor: Intents -> a smooth stream of (target, down, gripper, roll) -> the action model -> actions.

Every skill is a generator that yields one `Cmd` per 30 Hz tick, so any skill can be replaced mid-motion: a new
intent simply swaps the generator, and because targets reach the action model through a speed-limited "carrot",
redirects ("no, the blue one") bend the motion instead of jerking it. Nudges ("a bit left") shift whatever is
running. `stop` freezes in place; `resume` restarts the interrupted skill.

Works on the sim (object poses from physics) and on the real arm (object poses from a fixed layout); the tip
pose always comes from forward kinematics of the measured joints, never from privileged sim state.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from so101_sim2real.contract import load_contract
from so101_sim2real.sim import ArmSim

from .intent import Intent

import os
OPEN, CLOSED, HOLD_GRIP = float(os.environ.get("HACK_OPEN", 45.0)), 0.0, 0.0
GRASP_CLEAR = float(os.environ.get("HACK_CLEAR", 0.006))     # gap left between the fixed jaw and the cube going down
DESCEND_SPEED = float(os.environ.get("HACK_DESCEND", 0.12))
GRASP_DEPTH = float(os.environ.get("HACK_DEPTH", 0.010))     # tip below the cube centre at the grasp
LIFT_SPEED = float(os.environ.get("HACK_LIFT_SPEED", 0.30))   # m/s while lifting a fresh grasp
READY = np.array([0.20, 0.0, 0.17])
LIFT_Z = 0.14
ECHO_S = 6.0                 # a repeated identical intent within this many seconds of finishing is ignored
CARROT_SPEED = 0.30          # m/s of the target fed to the action model
WS_LO = np.array([0.08, -0.30, 0.008])
WS_HI = np.array([0.36, 0.30, 0.32])
PLACES = {"left": np.array([0.22, -0.16]), "right": np.array([0.22, 0.16]), "middle": np.array([0.24, 0.0]),
          "front": np.array([0.30, 0.0]), "back": np.array([0.15, 0.0])}
# viewer faces the arm (the demo camera is in front): the viewer's left is the robot's right (-y)
DIRVEC = {"left": np.array([0, -1.0, 0]), "right": np.array([0, 1.0, 0]), "up": np.array([0, 0, 1.0]),
          "down": np.array([0, 0, -1.0]), "forward": np.array([1.0, 0, 0]), "back": np.array([-1.0, 0, 0])}


@dataclass
class Cmd:
    target: np.ndarray
    down: float = 1.0
    grip: float | None = None
    roll: float | None = None          # owner-frame wrist_roll goal (deg); None = keep


@dataclass
class Status:
    skill: str = "idle"
    phase: str = ""
    holding: str | None = None
    target: list = field(default_factory=lambda: [0, 0, 0])
    tip: list = field(default_factory=lambda: [0, 0, 0])
    stopped: bool = False


class Kinematics:
    """FK / Jacobians of the nominal model from measured joints (what the real arm can compute too)."""

    def __init__(self):
        self.sim = ArmSim()
        self.m, self.d = self.sim.model, self.sim.data
        self.c = self.sim.contract
        self.tip = self.m.site("tip").id
        self.grip_body = self.m.body("gripper").id

    def set(self, joints):
        self.d.qpos[self.sim.qadr] = self.c.to_model(np.asarray(joints, float))
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def tip_pos(self, joints=None):
        if joints is not None:
            self.set(joints)
        return self.d.site_xpos[self.tip].copy()

    def finger_axis(self):
        return -self.d.xmat[self.grip_body].reshape(3, 3)[:, 2]     # fingers point along the gripper's -Z

    def gripper_yaw(self):
        x = self.d.xmat[self.grip_body].reshape(3, 3)[:, 0]
        return math.atan2(x[1], x[0])


class IKController:
    """Fallback with the ActionModel interface: full iterative IK each tick (position + fingers-down), then a
    rate-limited step of the joint goal toward the solution. Uses only measured joints and the nominal model."""

    def __init__(self, kin: Kinematics, max_delta: float = 6.0, iters: int = 12):
        self.k, self.max_delta, self.iters = kin, max_delta, iters
        self.q = None

    def reset(self):
        self.q = None

    def solve(self, start_owner, target, down):
        k = self.k
        m, d = k.m, k.d
        q = np.asarray(start_owner, float).copy()
        cols = k.sim.dadr[:4]
        lo, hi = k.c.low[:4] + 3, k.c.high[:4] - 3
        for _ in range(self.iters):
            k.set(q)
            Jp0 = np.zeros((3, m.nv)); Jr0 = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, Jp0, Jr0, k.tip)
            e = np.asarray(target) - d.site_xpos[k.tip]
            w = np.cross(k.finger_axis(), np.array([0, 0, -1.0]))
            free = np.ones(4, bool)
            for _pass in range(3):                     # active set: joints pinned at a limit drop out of the solve
                Jp, Jr = Jp0[:, cols] * free, Jr0[:, cols] * free
                Jinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + 1e-5 * np.eye(3))
                dq = Jinv @ e                                        # primary: tip position
                if down > 0.5:                                       # secondary: fingers down, in the null space
                    N = (np.eye(4) - Jinv @ Jp) * free[None, :] * free[:, None]
                    A = Jr @ N
                    dq = dq + N @ (A.T @ np.linalg.solve(A @ A.T + 1e-3 * np.eye(3), w - Jr @ dq))
                step = np.clip(np.rad2deg(dq), -15, 15)
                blocked = ((q[:4] >= hi - 1e-6) & (step > 0)) | ((q[:4] <= lo + 1e-6) & (step < 0))
                if not np.any(blocked & free):
                    break
                free &= ~blocked
            q[:4] = np.clip(q[:4] + step * free, lo, hi)
            if np.linalg.norm(e) < 5e-4 and down < 0.5:
                break
        return q

    def act(self, joints, prev_goal, target, down):
        prev_goal = np.asarray(prev_goal, float)
        seed = self.q if self.q is not None else np.asarray(joints, float)
        seed = seed.copy(); seed[4:] = np.asarray(joints, float)[4:]
        self.q = self.solve(seed, target, down)
        a = np.zeros(6)
        a[:4] = np.clip((self.q[:4] - prev_goal[:4]) / self.max_delta, -1, 1)
        return a


class Skills:
    def __init__(self, controller, kin: Kinematics | None = None, max_delta: float = 6.0):
        self.ctrl = controller
        self.kin = kin or Kinematics()
        self.c = load_contract()
        self.max_delta = max_delta
        self.status = Status()
        self.gen = None
        self.paused = None                 # (name, generator factory) interrupted by stop
        self.last_factory = None
        self.offset = np.zeros(3)
        self.carrot = None
        self.cmd = Cmd(READY.copy(), 1.0, 20.0, None)
        self.objects: dict = {}
        self.last_obj: str | None = None
        self.level_roll = float(self.c.start_pose[4])
        self.roll_sign = self._roll_sign()
        self.events: list[str] = []
        self._drop_ticks = 0
        self._last_key, self._last_done = None, -1e9
        self.origin: dict = {}                     # where each cube was picked from (to put it back)

    # ------------------------------------------------------------------ helpers
    def _roll_sign(self) -> float:
        """Does +roll (deg) turn the gripper's yaw + or -? Measured once on the nominal model, fingers down."""
        pose = np.array([0.0, -20.0, 40.0, 70.0, self.level_roll, 20.0])
        self.kin.set(pose); y0 = self.kin.gripper_yaw()
        pose[4] += 10; self.kin.set(pose); y1 = self.kin.gripper_yaw()
        return 1.0 if math.remainder(y1 - y0, 2 * math.pi) > 0 else -1.0

    def roll_for(self, yaw: float) -> float:
        """Wrist roll that lines the jaws up with a cube's faces (cube symmetry: 90 deg)."""
        rel = math.degrees(math.remainder(yaw, math.pi / 2))
        return self.level_roll + self.roll_sign * rel

    def align_roll(self, cube_yaw: float) -> float:
        """Roll goal that turns the jaws (gripper x) parallel to a cube face, measured against the gripper's
        CURRENT yaw (which includes pan), taking the smallest turn (cube symmetry: 90 deg)."""
        yaw_now = math.atan2(self._xg[1], self._xg[0])
        rel = math.degrees(math.remainder(cube_yaw - yaw_now, math.pi / 2))
        goal = self._joints[4] + self.roll_sign * rel
        lo, hi = self.c.low[4] + 5, self.c.high[4] - 5
        return float(np.clip(goal, lo, hi))

    def _in_bowl(self, name) -> bool:
        o, b = self.obj(name) if name else None, self.objects.get("bowl")
        return bool(o is not None and b is not None and np.linalg.norm(o["pos"][:2] - b["pos"][:2]) < 0.06)

    def obj(self, name):
        return self.objects.get(name)

    def _say(self, msg):
        self.events.append(msg)

    # ------------------------------------------------------------------ intents
    def handle(self, it: Intent, last_obj: str | None = None) -> None:
        self.last_obj = last_obj or self.last_obj
        k = it.kind
        if k == "stop":
            if self.gen is not None:
                self.paused = self.last_factory
            self.gen = None
            self.status.skill, self.status.phase, self.status.stopped = "stop", "holding still", True
            self.cmd = Cmd(self.kin.tip_pos(self._joints), self.cmd.down, self.cmd.grip, self.cmd.roll)
            self.carrot = self.cmd.target.copy()
            self.offset[:] = 0
            return
        self.status.stopped = False
        if k == "resume":
            if self.paused:
                self._start(*self.paused)
            return
        if k in ("nudge", "move"):
            v = DIRVEC[it.dir] * (it.amount or 0.03)
            if self.gen is None:
                self.cmd.target = np.clip(self.cmd.target + v, WS_LO, WS_HI)
                self.status.skill, self.status.phase = it.kind, it.dir
            else:
                self.offset += v
            return
        factories = {
            "pick": lambda: self.pick(it.obj), "point": lambda: self.point(it.obj),
            "place": lambda: self.place(it.where), "home": self.home, "wave": self.wave, "nod": self.nod,
            "dance": self.dance, "open": lambda: self.set_grip(OPEN, "open"), "close": lambda: self.set_grip(CLOSED, "close"),
            "cleanup": self.cleanup, "tower": self.tower, "lineup": self.lineup,
        }
        if k in factories:
            key = (k, it.obj, it.where)
            # PersonaPlex often confirms twice ("Putting it in the bowl. Okay, putting it in the bowl."):
            # the same intent while it is running, or within ECHO_S of finishing it, is an echo, not a new request
            if key == self._last_key and (self.gen is not None or time.perf_counter() - self._last_done < ECHO_S):
                return
            if k == "pick" and it.obj == self.status.holding:
                return
            if k == "pick" and self._in_bowl(it.obj) and self._last_key == ("place", None, "bowl"):
                return
            self._last_key = key
            if k == "place" and self.status.holding is None and it.where == "bowl" and self._in_bowl(self.last_obj):
                return                                             # already there: nothing to do
            if k == "place" and self.status.holding is None and self.last_obj and self.last_obj != it.where:
                obj = self.last_obj               # "put the red cube in the bowl" without a pick first
                self._start(f"place {it.where}", lambda: self.chain(self.pick(obj, settle=False), self.place(it.where)))
                return
            name = f"{k} {it.obj or it.where or ''}".strip()
            self._start(name, factories[k])

    def _start(self, name, factory):
        self.last_factory = (name, factory)
        self.offset[:] = 0
        self.gen = factory()
        self.status.skill, self.status.phase = name, ""

    @staticmethod
    def chain(*gens):
        for g in gens:
            yield from g

    # ------------------------------------------------------------------ skills
    def _go(self, target_fn, down=1.0, grip=None, roll=None, tol=0.012, timeout=4.0, phase=""):
        """Yield commands until the tip is within tol of the (possibly moving) target, or timeout."""
        n = int(timeout * 30)
        for _ in range(n):
            t = np.asarray(target_fn(), float)
            self.status.phase = phase
            yield Cmd(t, down, grip, roll() if callable(roll) else roll)
            if np.linalg.norm(self._tip - (t + self.offset)) < tol:
                return

    def _wait(self, secs, grip=None, phase="", down=1.0):
        t = self.cmd.target.copy()
        for _ in range(int(secs * 30)):
            self.status.phase = phase
            yield Cmd(t, down, grip, self.cmd.roll)

    def pick(self, name, settle=True):
        if self.obj(name) is None:
            return
        half = self.obj(name)["size"] / 2
        # the tip site sits on the FIXED jaw's inner face; the cube has to sit between the jaws, i.e. half a cube
        # further along the gripper's +x (toward the moving jaw)
        o = lambda: self.obj(name)["pos"] - (half + GRASP_CLEAR) * self._xg
        roll = lambda: self.align_roll(self.obj(name)["yaw"])
        if self.status.holding and self.status.holding != name:     # changed its mind: put the first one back
            yield from self.place_at(self.origin.get(self.status.holding, self._tip))
        yield from self._go(lambda: o() + [0, 0, 0.09], grip=OPEN, roll=roll, tol=0.02, phase="approach")
        yield from self._go(lambda: np.r_[o()[:2], max(0.004, o()[2] - GRASP_DEPTH)], grip=OPEN, roll=roll, tol=0.008,
                            timeout=2.5, phase="descend")
        self.origin[name] = self.obj(name)["pos"].copy()
        yield from self._wait(0.6, grip=CLOSED, phase="grasp")
        self.status.holding = name
        self._say(f"holding {name}")
        yield from self._go(lambda: np.r_[self.cmd.target[:2], LIFT_Z], grip=CLOSED, tol=0.02, phase="lift")
        if settle:
            yield from self._wait(0.3, grip=CLOSED, phase="holding")

    def _held_offset(self):
        """Where the held cube sits relative to the tip (measured, so placements land the CUBE, not the tip)."""
        h = self.status.holding
        if h and self.obj(h) is not None:
            return self.obj(h)["pos"] - self._tip, self.obj(h)["size"] / 2
        return np.zeros(3), 0.0

    def place_at(self, xy, into_bowl=False, on_obj=None):
        hover_z = LIFT_Z + (0.07 if on_obj else 0.03)
        dest = lambda: np.asarray(self.obj(on_obj)["pos"] if on_obj else xy, float)[:2]
        off = lambda: self._held_offset()[0][:2]
        if into_bowl:
            low = 0.11
        elif on_obj:                                # held cube's bottom 4 mm above the support's top
            sup = self.obj(on_obj)
            low = lambda: sup["pos"][2] + sup["size"] / 2 + 0.004 - self._held_offset()[0][2] + self._held_offset()[1]
        else:
            low = lambda: 0.004 - self._held_offset()[0][2] + max(self._held_offset()[1], 0.015)
        lowz = low if callable(low) else (lambda: low)
        yield from self._go(lambda: np.r_[dest() - off(), hover_z], grip=CLOSED, tol=0.015, phase="carry")
        yield from self._go(lambda: np.r_[dest() - off(), lowz()], grip=CLOSED, tol=0.008, timeout=3.0, phase="lower")
        yield from self._wait(0.5, grip=OPEN, phase="release")
        if self.status.holding:
            self._say(f"released {self.status.holding}")
        self.status.holding = None
        up = np.r_[self._tip[:2], hover_z]
        yield from self._go(lambda: up, grip=OPEN, tol=0.02, phase="retreat")

    # ------------------------------------------------------------------ multi-step tasks
    def cubes(self):
        return [c for c in ("red", "green", "blue") if self.obj(c) is not None]

    def cleanup(self):
        """Everything into the bowl, nearest first."""
        for _ in range(3):
            todo = [c for c in self.cubes() if not self._in_bowl(c)]
            if not todo:
                break
            c = min(todo, key=lambda c: np.linalg.norm(self.obj(c)["pos"][:2] - self._tip[:2]))
            self.status.skill = f"clean up: {c}"
            yield from self.pick(c, settle=False)
            yield from self.place_at(self.objects["bowl"]["pos"], into_bowl=True)
        yield from self.home()

    def tower(self):
        """Stack all three: the one nearest the middle is the base."""
        free = [c for c in self.cubes() if not self._in_bowl(c)]
        if len(free) < 2:
            return
        free.sort(key=lambda c: abs(self.obj(c)["pos"][1]))
        base = free[0]
        for c in free[1:]:
            self.status.skill = f"tower: {c} on {base}"
            yield from self.pick(c, settle=False)
            yield from self.place_at(None, on_obj=base)
            base = c
        yield from self.home()

    def lineup(self):
        """Red, green, blue in a row across the front of the table (viewer's left to right)."""
        slots = {"red": np.array([0.29, -0.08]), "green": np.array([0.29, 0.0]), "blue": np.array([0.29, 0.08])}
        for c in self.cubes():
            if self._in_bowl(c) or np.linalg.norm(self.obj(c)["pos"][:2] - slots[c]) < 0.02:
                continue
            self.status.skill = f"line up: {c}"
            yield from self.pick(c, settle=False)
            yield from self.place_at(slots[c])
        yield from self.home()

    def place(self, where):
        if where == "bowl":
            yield from self.place_at(self.objects["bowl"]["pos"], into_bowl=True)
        elif where in ("red", "blue", "green") and self.obj(where) is not None:
            yield from self.place_at(None, on_obj=where)
        elif where in PLACES:
            yield from self.place_at(PLACES[where])
        else:
            yield from self.place_at(self._tip)

    def point(self, name):
        if self.obj(name) is None:
            return
        yield from self._go(lambda: self.obj(name)["pos"] + [0, 0, 0.10], grip=CLOSED if not self.status.holding else None,
                            tol=0.015, timeout=4.0, phase="pointing")
        yield from self._wait(1.0, phase="pointing")

    def home(self):
        yield from self._go(lambda: READY, grip=20.0 if not self.status.holding else None, roll=self.level_roll,
                            phase="home", timeout=4.0)

    def set_grip(self, value, name):
        if value == OPEN and self.status.holding:
            self._say(f"released {self.status.holding}")
            self.status.holding = None
        yield from self._wait(0.5, grip=value, phase=name)

    def wave(self):
        up = np.array([0.20, 0.0, 0.27])
        yield from self._go(lambda: up, down=0.0, grip=OPEN if not self.status.holding else None, tol=0.03, phase="raise")
        for i in range(int(2.4 * 30)):
            s = math.sin(2 * math.pi * 1.2 * i / 30)
            yield Cmd(up + [0, 0.07 * s, 0], 0.0, None, self.level_roll + 25 * s)
        yield from self.home()

    def nod(self):
        base = self._tip.copy()
        for i in range(int(1.6 * 30)):
            yield Cmd(base + [0, 0, 0.03 * math.sin(2 * math.pi * 1.25 * i / 30)], 1.0, None, None)

    def dance(self):
        c = np.array([0.22, 0.0, 0.20])
        for i in range(int(4 * 30)):
            t = i / 30
            yield Cmd(c + [0.03 * math.sin(2 * math.pi * 0.5 * t), 0.08 * math.sin(2 * math.pi * 1.0 * t),
                           0.04 * math.sin(2 * math.pi * 2.0 * t)], 0.0, OPEN if int(t * 2) % 2 else 20.0,
                      self.level_roll + 30 * math.sin(2 * math.pi * 1.0 * t))
        yield from self.home()

    # ------------------------------------------------------------------ the 30 Hz tick
    def tick(self, obs: dict, objects: dict) -> np.ndarray:
        self._joints = np.asarray(obs["joints"], float)
        self._tip = self.kin.tip_pos(self._joints)
        xg = self.kin.d.xmat[self.kin.grip_body].reshape(3, 3)[:, 0].copy()
        xg[2] = 0.0
        self._xg = xg / (np.linalg.norm(xg) + 1e-9)            # gripper x (fixed -> moving jaw), horizontal
        self.objects = objects
        if self.carrot is None:
            self.carrot = self._tip.copy()
        if self.gen is not None:
            try:
                self.cmd = next(self.gen)
            except StopIteration:
                self.gen = None
                self._last_done = time.perf_counter()
                done = self.status.skill
                self.status.skill, self.status.phase = "idle", f"done: {done}"
                self.cmd = Cmd(self.cmd.target.copy(), self.cmd.down, self.cmd.grip, self.cmd.roll)
        # holding check: a cube that is no longer between the fingers was dropped
        h = self.status.holding
        if h and self.obj(h) is not None and self.status.phase not in ("grasp", "release"):
            grasp_pt = self._tip + (self.obj(h)["size"] / 2 + 0.002) * self._xg
            far = np.linalg.norm(self.obj(h)["pos"] - grasp_pt) > 0.045
            self._drop_ticks = self._drop_ticks + 1 if far else 0
            if self._drop_ticks >= 8:                      # ~0.27 s away from the fingers: it really fell
                self._say(f"dropped {h}")
                self.status.holding = None
                self._drop_ticks = 0
        goal = np.clip(self.cmd.target + (self.offset if self.gen is not None else 0), WS_LO, WS_HI)
        step = goal - self.carrot
        dist = np.linalg.norm(step)
        vmax = {"lift": LIFT_SPEED, "descend": DESCEND_SPEED, "lower": DESCEND_SPEED}.get(self.status.phase, CARROT_SPEED) / 30
        self.carrot = goal if dist <= vmax else self.carrot + step / dist * vmax
        a = np.asarray(self.ctrl.act(self._joints, obs["prev_goal"], self.carrot, self.cmd.down), float).copy()
        prev = np.asarray(obs["prev_goal"], float)
        if self.cmd.roll is not None:
            a[4] = np.clip((self.cmd.roll - prev[4]) / self.max_delta, -1, 1)
        else:
            a[4] = 0.0
        if self.cmd.grip is not None:
            a[5] = np.clip((self.cmd.grip - prev[5]) / self.max_delta, -1, 1)
        else:
            a[5] = 0.0
        self.status.target = [round(float(x), 3) for x in self.carrot]
        self.status.tip = [round(float(x), 3) for x in self._tip]
        return a
