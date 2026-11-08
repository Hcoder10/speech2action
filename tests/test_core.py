import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from so101_sim2real.contract import goal_from_action, load_contract, preprocess_image
from so101_sim2real.env import DRConfig, SO101Env
from so101_sim2real.servo import ParamRanges, prior
from so101_sim2real.sim import ArmSim
from so101_sim2real.sysid.excite import Segment, build_program, check_program, find_home
from so101_sim2real.sysid.fit import FIT, fit_joint
from so101_sim2real.sysid.replay import make_sim, replay_segment, synthesize

warnings.filterwarnings("ignore", message="no assets/fitted.json")


def test_contract_roundtrip_and_ticks():
    c = load_contract()
    pose = np.array([10.0, -20.0, 30.0, 40.0, 75.6, 33.0])
    assert np.allclose(c.from_model(c.to_model(pose)), pose)
    q = c.quantize(pose)
    assert np.all(np.abs(q - pose)[:5] <= 360 / 4096 / 2 + 1e-9)


def test_preprocess_letterboxes_4x3():
    img = np.full((480, 640, 3), 200, np.uint8)
    out = preprocess_image(img, (128, 128))
    assert out.shape == (128, 128, 3)
    assert out[0, 64].sum() == 0 and out[64, 64].sum() > 0     # black bar on top, image in the middle


def test_goal_from_action_matches_lerobot_clamp():
    c = load_contract()
    read = np.array([0.0, -50, 35, 60, 75.6, 20])
    g = goal_from_action(np.ones(6), "delta", read, read, c, max_delta=30.0, max_relative_target=10.0)
    assert np.allclose(g - read, 10.0)


def test_servo_step_response_is_speed_limited():
    sim = ArmSim(prior())
    home = np.array([0.0, -50, 35, 60, 75.6, 20])
    sim.reset(home); sim.advance_to(0.5)
    g = home.copy(); g[0] += 40
    sim.command(g); sim.advance_to(0.5 + 1 / 30)
    first = sim.read()[0] - home[0]
    sim.advance_to(1.5)
    assert first < 6.0                                  # latency + finite speed: no teleporting servo
    assert abs(sim.read()[0] - g[0]) < 0.5              # but it gets there


@pytest.mark.parametrize("keys", [("joints", "prev_goal", "target"), ("joints", "wrist")])
def test_env_api(keys):
    env = SO101Env("reach" if "target" in keys else "lift", obs_keys=keys, seed=0, image_size=(64, 64))
    check_env(env, skip_render_check=True)


def test_env_determinism():
    def roll(seed):
        env = SO101Env("lift", seed=0)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(1)
        out = [obs["joints"]]
        for _ in range(20):
            obs, *_ = env.step(rng.uniform(-1, 1, 6))
            out.append(obs["joints"])
        return np.array(out)
    assert np.array_equal(roll(5), roll(5))
    assert not np.array_equal(roll(5), roll(6))


def test_dr_off_is_nominal_and_on_perturbs_calibration():
    env = SO101Env("reach", dr=DRConfig.off(), seed=0)
    _, info = env.reset(seed=0)
    assert np.allclose(info["privileged"]["offset_err"], 0)
    assert env.ep_params.latency == env.nominal_params.latency
    env = SO101Env("reach", seed=0)
    offs = [env.reset(seed=s)[1]["privileged"]["offset_err"] for s in range(8)]
    assert np.std(np.array(offs)[:, 1]) > 0.5


def test_observation_is_true_pose_through_calibration_error():
    env = SO101Env("reach", dr=DRConfig(encoder_noise_ticks=0.0), seed=0)
    obs, info = env.reset(seed=3)
    p = info["privileged"]
    assert np.allclose(obs["joints"], p["true_joints"] - p["offset_err"], atol=0.1)


def test_scene_props_at_rest():
    for task in ("reach", "lift"):
        env = SO101Env(task, seed=0)
        env.reset(seed=1)
        for _ in range(30):
            env.step(np.zeros(6))
        assert np.abs(env.sim.data.qvel).max() < 0.05


def test_safety_check_rejects_table_strike():
    sim = ArmSim(prior())
    home = find_home(sim)
    bad = home.copy(); bad[1] += 60; bad[2] += 40                       # lift + elbow down: into the table
    seg = Segment("bad", 1, np.array([0.0, 1 / 30]), np.array([home, bad]))
    assert check_program(sim, [seg])
    assert not check_program(sim, build_program(home, sim.contract.low, sim.contract.high, seed=1)[:1])


def test_dr_cov_sampling_stays_near_fit():
    p = prior()
    cov = [np.diag(np.full(len(FIT), 0.01)) for _ in range(6)]
    r = ParamRanges.around(p); r.cov = cov
    rng = np.random.default_rng(0)
    ratios = np.array([r.sample(p, rng).kp / p.kp for _ in range(50)])
    assert ratios.min() > np.exp(-0.2 - np.log(1.1)) - 1e-9 and ratios.max() < np.exp(0.2 + np.log(1.1)) + 1e-9


def test_sysid_recovers_pan_and_latency():
    """Fake arm with a 20x softer, slower servo and 35 ms latency; the pan+latency fit must recover it."""
    truth = prior()
    truth.kp = truth.kp * 0.05; truth.tau = truth.tau * 0.8; truth.latency = 0.035
    sim = make_sim(prior())
    home = find_home(sim)
    prog = build_program(home, sim.contract.low, sim.contract.high, seed=3, joints=[0])
    rec = synthesize(truth, home, prog, seed=0)
    r = fit_joint((prior(), rec, 0, True, 25))
    assert r["rmse_after"] < 0.15 < r["rmse_before"]
    assert abs(r["x"][len(FIT)] / 100 - 0.035) < 0.006
    fitted = prior()
    fitted.kp[0], fitted.tau[0] = np.exp(r["x"][0]), np.exp(r["x"][2])
    assert 0.5 < fitted.kp[0] / truth.kp[0] < 2.0
