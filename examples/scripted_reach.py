"""Solve `reach` with a controller that sees only what a policy sees (joints + target) and the NOMINAL model.

Damped-least-squares IK on the nominal kinematics, emitted as delta actions. Under full randomisation its final
error measures what calibration error, latency and servo spread cost a policy that never adapts.
  python examples/scripted_reach.py [--episodes 20] [--dr 1.0]
"""
import argparse

import mujoco
import numpy as np

from so101_sim2real.contract import load_contract
from so101_sim2real.env import DRConfig, SO101Env
from so101_sim2real.sim import ArmSim


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=20); ap.add_argument("--dr", type=float, default=1.0)
    a = ap.parse_args()
    env = SO101Env("reach", dr=DRConfig(scale=a.dr), seed=0)
    kin = ArmSim(); c = load_contract(); m, d = kin.model, kin.data
    tip = m.site("tip").id
    errs, succ = [], 0
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(int(env.task.episode_seconds * 30)):
            d.qpos[kin.qadr] = c.to_model(obs["joints"]); mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            J = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, J, None, tip)
            J = J[:, kin.dadr[:4]]                                   # pan, lift, elbow, wrist_flex
            above = obs["target"] + np.array([0, 0, 0.05])            # hover first, then descend
            horiz = np.linalg.norm((obs["target"] - d.site_xpos[tip])[:2])
            goal = above if horiz > 0.01 else obs["target"]
            e = goal - d.site_xpos[tip]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), np.clip(0.3 * e, -0.01, 0.01))   # gain 0.3: ~2 ticks of servo+bus delay
            act = np.zeros(6); act[:4] = np.clip(np.rad2deg(dq) / env.max_delta, -1, 1)
            obs, r, term, trunc, info = env.step(act)
        errs.append(info["distance"]); succ += info["success"]
    errs = np.array(errs) * 1000
    print(f"dr={a.dr}: final tip error median {np.median(errs):.1f} mm, p90 {np.percentile(errs, 90):.1f} mm, success {succ}/{a.episodes}")


if __name__ == "__main__":
    main()
