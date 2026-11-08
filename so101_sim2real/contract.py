"""The I/O contract shared by the simulator and the real arm.

Everything a policy sees or sends is in the units LeRobot 0.6.1 `so_follower(use_degrees=True)` uses on the
owner's arm: five joint angles in degrees (owner's calibration zeros) and the gripper in 0..100. The simulator
converts to model radians internally through the bench-fitted offsets in `assets/calibration.json`, so a policy
never sees a model-frame number and cannot learn one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = JOINTS[:5]
CONTROL_HZ = 30.0
TICKS_PER_REV = 4096                      # STS3215 magnetic encoder
DEG_PER_TICK = 360.0 / TICKS_PER_REV


@dataclass(frozen=True)
class Contract:
    offset_deg: np.ndarray                # model_deg = owner_deg + offset (arm joints; gripper entry unused)
    jaw_closed_rad: float
    jaw_span_rad: float
    low: np.ndarray                       # owner-frame limits, degrees / gripper 0..100
    high: np.ndarray
    start_pose: np.ndarray                # sag-safe folded pose recorded on the bench (owner frame)
    gripper_deg_per_unit: float           # gripper 0..100 -> joint degrees, for tick quantisation

    def to_model(self, owner: np.ndarray) -> np.ndarray:
        """Owner degrees / gripper units -> model joint radians. Works on (..., 6)."""
        owner = np.asarray(owner, float)
        out = np.empty_like(owner)
        out[..., :5] = np.deg2rad(owner[..., :5] + self.offset_deg[:5])
        out[..., 5] = self.jaw_closed_rad + owner[..., 5] / 100.0 * self.jaw_span_rad
        return out

    def from_model(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, float)
        out = np.empty_like(qpos)
        out[..., :5] = np.rad2deg(qpos[..., :5]) - self.offset_deg[:5]
        out[..., 5] = (qpos[..., 5] - self.jaw_closed_rad) / self.jaw_span_rad * 100.0
        return out

    def vel_to_model(self, owner_vel: np.ndarray) -> np.ndarray:
        owner_vel = np.asarray(owner_vel, float)
        out = np.empty_like(owner_vel)
        out[..., :5] = np.deg2rad(owner_vel[..., :5])
        out[..., 5] = owner_vel[..., 5] / 100.0 * self.jaw_span_rad
        return out

    def clip(self, owner: np.ndarray) -> np.ndarray:
        return np.clip(owner, self.low, self.high)

    def quantize(self, owner: np.ndarray) -> np.ndarray:
        """Round to whole encoder ticks the way Present_Position does."""
        owner = np.asarray(owner, float)
        step = np.r_[np.full(5, DEG_PER_TICK), DEG_PER_TICK / self.gripper_deg_per_unit]
        return np.round(owner / step) * step


def load_contract(path: Path | None = None) -> Contract:
    cal = json.loads(Path(path or ASSETS / "calibration.json").read_text())
    conv = cal["joint_convention"]
    rng = cal["owner_range_deg"]
    offset = np.array([conv["offset_deg"][j] for j in ARM] + [0.0])
    low = np.array([rng[j][0] for j in ARM] + [0.0])
    high = np.array([rng[j][1] for j in ARM] + [100.0])
    return Contract(
        offset_deg=offset,
        jaw_closed_rad=float(conv["jaw_closed_rad"]),
        jaw_span_rad=float(conv["jaw_span_rad"]),
        low=low,
        high=high,
        start_pose=np.array(cal["real_start_pose_deg"], float),
        gripper_deg_per_unit=float(np.rad2deg(conv["jaw_span_rad"]) / 100.0),
    )


def preprocess_image(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The ONE image transform used by both sim and real: letterbox the 4:3 camera frame to a square (black bars,
    whole field of view kept, aspect preserved), then area-resize to `size` (w, h). uint8 RGB in and out."""
    import cv2

    h, w = rgb.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    return np.ascontiguousarray(cv2.resize(canvas, tuple(size), interpolation=cv2.INTER_AREA))


def goal_from_action(action, mode: str, prev_goal, last_read, contract: Contract, max_delta: float,
                     max_relative_target: float | None) -> np.ndarray:
    """Policy action in [-1, 1]^6 -> owner-frame goal. Shared verbatim by the sim env and the real twin.

    abs:   action spans each joint's owner range.
    delta: goal = previous goal + action * max_delta (degrees per 30 Hz step; gripper units).
    max_relative_target clamps the goal around the MEASURED position, exactly as LeRobot's send_action does.
    """
    a = np.clip(np.asarray(action, float), -1, 1)
    if mode == "abs":
        goal = contract.low + (a + 1) / 2 * (contract.high - contract.low)
    elif mode == "delta":
        goal = np.asarray(prev_goal, float) + a * max_delta
    else:
        raise ValueError(mode)
    if max_relative_target is not None:
        goal = np.clip(goal, last_read - max_relative_target, last_read + max_relative_target)
    return contract.clip(goal)
