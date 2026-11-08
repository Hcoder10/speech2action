"""Thin hardware layer over LeRobot 0.6.1 `so_follower` (owner's bench: COM3, id follower_1, degrees).

Only this module imports lerobot, so the simulator never needs it. Run hardware scripts with the LeRobot venv:
  C:/Users/sarta/reachy-duplex/.venv-so101/Scripts/python.exe -m so101_sim2real.sysid.record ...
(from the repo root, so the package resolves without installing).

Safety rules that are enforced here, from the bench sessions of Sep 2026:
  * nothing moves unless the caller passes allow_motion=True (the owner is at the arm);
  * an abort sets goal := measured pose with torque ON; it never drops torque (a dropped arm falls);
  * a file named STOP in the run directory aborts at the next tick (an agent cannot type into a prompt);
  * moves to a pose are joint-interpolated at <= 15 deg/s and checked in the sim first by the caller.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .contract import JOINTS

DEFAULT_PORT = "COM3"
DEFAULT_ID = "follower_1"
CAL_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"


class Arm:
    def __init__(self, port: str = DEFAULT_PORT, robot_id: str = DEFAULT_ID, allow_motion: bool = False,
                 stop_file: Path | None = None):
        from lerobot.robots.so_follower import SOFollower, SOFollowerConfig

        cfg = SOFollowerConfig(port=port, cameras={}, use_degrees=True, max_relative_target=None)
        cfg.id = robot_id                       # 0.6.1 rejects id= in the constructor
        cfg.calibration_dir = CAL_DIR
        if hasattr(cfg, "disable_torque_on_disconnect"):
            cfg.disable_torque_on_disconnect = False
        self.robot = SOFollower(cfg)
        self.robot.connect(calibrate=False)     # writes P=16 I=0 D=32 (the deployment gains)
        self.bus = self.robot.bus
        self.allow_motion = allow_motion
        self.stop_file = stop_file
        self.gains = {"P": cfg.position_p_coefficient, "I": cfg.position_i_coefficient, "D": cfg.position_d_coefficient}

    def read(self) -> np.ndarray:
        v = self.bus.sync_read("Present_Position")
        return np.array([v[j] for j in JOINTS], float)

    def read_load(self) -> np.ndarray:
        v = self.bus.sync_read("Present_Load", normalize=False)
        raw = np.array([v[j] for j in JOINTS], float)
        return np.where(raw >= 1024, -(raw - 1024), raw)    # bit 10 is the direction bit

    def write(self, goal: np.ndarray) -> None:
        if not self.allow_motion:
            raise PermissionError("motion not allowed: pass allow_motion=True only with the owner at the arm")
        self.bus.sync_write("Goal_Position", {j: float(g) for j, g in zip(JOINTS, goal)})

    def stop_requested(self) -> bool:
        return bool(self.stop_file and Path(self.stop_file).exists())

    def hold(self) -> np.ndarray:
        """Abort: goal := where the arm is now, torque stays on."""
        pos = self.read()
        if self.allow_motion:
            self.write(pos)
        return pos

    def move_to(self, pose: np.ndarray, speed_deg_s: float = 15.0, hz: float = 30.0) -> np.ndarray:
        start = self.read()
        span = np.max(np.abs(np.asarray(pose) - start)[:5]) if len(pose) else 0
        n = max(2, int(np.ceil(span / speed_deg_s * hz)))
        t0 = time.perf_counter()
        for k in range(1, n + 1):
            if self.stop_requested():
                return self.hold()
            self.write(start + (np.asarray(pose) - start) * k / n)
            while time.perf_counter() < t0 + k / hz:
                time.sleep(0.001)
        time.sleep(0.5)
        return self.read()

    def close(self) -> None:
        """Release the port with torque left ON (the arm keeps holding its last goal)."""
        try:
            self.bus.port_handler.closePort()
        except Exception:
            pass


def open_wrist_camera(index: int = 1, width: int = 640, height: int = 480):
    """Innomaker U20CAM-720P on DirectShow, 640x480 = the centre 4:3 crop the sim camera was fitted to."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} did not open")
    return cap


def grab_rgb(cap) -> np.ndarray:
    import cv2

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera read failed")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
