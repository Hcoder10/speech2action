"""The numpy ActionModel reproduces SB3's deterministic prediction (needs the training run that produced reacher.npz).

  .venv/Scripts/python.exe -m pytest -q hack/action_model/test_model.py
"""
from pathlib import Path

import numpy as np
import pytest

from hack.action_model.kin import Kin, chain_from_model, features
from hack.action_model.model import DEFAULT, ActionModel

HERE = Path(__file__).resolve().parent


def _matching_run(am: ActionModel):
    from stable_baselines3.common.vec_env import VecNormalize
    import pickle
    for vn in sorted(HERE.glob("runs/*/*_vecnorm.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(vn, "rb") as f:
            v = pickle.load(f)
        if v.obs_rms.mean.shape == am.mean.shape and np.allclose(v.obs_rms.mean, am.mean, atol=0, rtol=0):
            return vn.with_name(vn.name.replace("_vecnorm.pkl", ".zip")), v
    return None, None


def test_kin_matches_mujoco():
    import mujoco
    from so101_sim2real.contract import load_contract
    from so101_sim2real.sim import ArmSim
    k, s, c = Kin(chain_from_model()), ArmSim(), load_contract()
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(c.low, c.high)
        s.data.qpos[s.qadr] = c.to_model(p); mujoco.mj_kinematics(s.model, s.data)
        tip, R = k.fk(p)
        assert np.allclose(tip, s.data.site("tip").xpos, atol=1e-9)
        assert np.allclose(R, s.data.body("gripper").xmat.reshape(3, 3), atol=1e-9)


@pytest.mark.skipif(not DEFAULT.exists(), reason="no reacher.npz exported yet")
def test_numpy_matches_sb3():
    from stable_baselines3 import PPO
    am = ActionModel()
    run, vn = _matching_run(am)
    if run is None:
        pytest.skip("training run for this reacher.npz is not on disk")
    model = PPO.load(run, device="cpu")
    rng = np.random.default_rng(1)
    for _ in range(200):
        joints = rng.uniform(-100, 100, 6); joints[5] = rng.uniform(0, 100)
        goal = joints + rng.uniform(-8, 8, 6)
        target = rng.uniform([0.05, -0.3, 0.0], [0.4, 0.3, 0.3])
        down = float(rng.integers(2))
        obs = features(am.kin, joints, goal, target, down, am.feat_version)
        ref, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        out = am.act(joints, goal, target, down)
        assert out.shape == (6,) and out[4] == 0 and out[5] == 0
        np.testing.assert_allclose(out[:4], ref[0], atol=1e-5)
