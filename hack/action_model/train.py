"""PPO for the reacher. 20 SubprocVecEnv workers, MLP 256x256, CPU.

  .venv/Scripts/python.exe -m hack.action_model.train --minutes 40
Logs, checkpoints and the best model (by periodic deterministic eval) go to hack/action_model/runs/<name>/;
the best one is also exported to runs/<name>/reacher.npz every time it improves (copy it to hack/action_model/).
"""
from __future__ import annotations

import argparse
import json
import os
import time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # 20 workers x 24 BLAS threads exhausted
    os.environ.setdefault(_k, "1")                                         # the commit charge (1.2 GB/worker)
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from .kin import FEAT_VERSION, chain_from_model
from .train_env import ReachEnv, make_env

HERE = Path(__file__).resolve().parent


def export_npz(model: PPO, vecnorm: VecNormalize, path: Path, meta: dict | None = None) -> None:
    pol = model.policy
    lin = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    out = {}
    for i, l in enumerate(lin):
        out[f"W{i}"] = l.weight.detach().cpu().numpy().astype(np.float64)
        out[f"b{i}"] = l.bias.detach().cpu().numpy().astype(np.float64)
    out["Wa"] = pol.action_net.weight.detach().cpu().numpy().astype(np.float64)
    out["ba"] = pol.action_net.bias.detach().cpu().numpy().astype(np.float64)
    out["n_hidden"] = np.array(len(lin))
    out["obs_mean"] = vecnorm.obs_rms.mean.astype(np.float64)
    out["obs_var"] = vecnorm.obs_rms.var.astype(np.float64)
    out["obs_eps"] = np.array(vecnorm.epsilon)
    out["obs_clip"] = np.array(vecnorm.clip_obs)
    out.update({k: np.asarray(v) for k, v in chain_from_model().items()})
    out["meta"] = np.array(json.dumps(meta or {}))
    out["feat_version"] = np.array(FEAT_VERSION)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)


def evaluate(model: PPO, vecnorm: VecNormalize, env: ReachEnv, episodes: int, seed: int) -> dict:
    errs, succ, aligns, contacts = [], [], [], 0
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a, _ = model.predict(vecnorm.normalize_obs(o[None]), deterministic=True)
            o, r, te, tr, info = env.step(a[0])
            contacts += info["contact"]
            done = te or tr
        errs.append(info["dist"]); succ.append(info["success"])
        if info["down"]:
            aligns.append(info["align"])
    errs = np.array(errs) * 1000
    res = {"median_mm": float(np.median(errs)), "p90_mm": float(np.percentile(errs, 90)),
           "success": float(np.mean(succ)), "down_align": float(np.mean(aligns)) if aligns else None,
           "contact_steps": int(contacts)}
    res["score"] = res["success"] - res["median_mm"] / 100.0
    return res


class Watch(BaseCallback):
    def __init__(self, run: Path, vecnorm: VecNormalize, deadline: float, eval_every_s: float, ckpt_every_s: float):
        super().__init__()
        self.run, self.vecnorm, self.deadline = run, vecnorm, deadline
        self.eval_every_s, self.ckpt_every_s = eval_every_s, ckpt_every_s
        self.t0 = time.time(); self.last_eval = self.t0; self.last_ckpt = self.t0
        self.best = -1e9
        self.eval_env = ReachEnv(seed=12345)
        self.log = open(run / "eval.jsonl", "a")

    def _on_step(self) -> bool:
        return time.time() < self.deadline

    def _on_rollout_end(self) -> None:
        now = time.time()
        if now - self.last_ckpt > self.ckpt_every_s:
            self.model.save(self.run / "last.zip"); self.vecnorm.save(str(self.run / "last_vecnorm.pkl"))
            self.last_ckpt = now
        if now - self.last_eval > self.eval_every_s or now > self.deadline:
            self.last_eval = now
            res = evaluate(self.model, self.vecnorm, self.eval_env, 40, seed=10_000)
            res.update(steps=int(self.num_timesteps), minutes=round((now - self.t0) / 60, 2))
            better = res["score"] > self.best
            res["best"] = better
            print("EVAL", json.dumps(res), flush=True)
            self.log.write(json.dumps(res) + "\n"); self.log.flush()
            if better:
                self.best = res["score"]
                self.model.save(self.run / "best.zip"); self.vecnorm.save(str(self.run / "best_vecnorm.pkl"))
                export_npz(self.model, self.vecnorm, self.run / "reacher.npz", meta=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--name", default=time.strftime("ppo_%H%M"))
    ap.add_argument("--eval-every", type=float, default=150, help="seconds")
    ap.add_argument("--resume", default=None, help="run dir to continue from (uses last.zip)")
    ap.add_argument("--export", nargs=2, metavar=("CKPT_ZIP", "OUT_NPZ"), default=None,
                    help="export a checkpoint (its <stem>_vecnorm.pkl beside it) to a numpy npz and exit")
    a = ap.parse_args()
    if a.export:
        z = Path(a.export[0])
        vn = VecNormalize.load(str(z.with_name(z.stem + "_vecnorm.pkl")), DummyVecEnv([make_env(0)]))
        export_npz(PPO.load(z, device="cpu"), vn, Path(a.export[1]), meta={"ckpt": str(z)})
        print("exported", a.export[1])
        return
    torch.set_num_threads(6)
    run = HERE / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    venv = VecMonitor(SubprocVecEnv([make_env(i, seed=1000 * i + 7) for i in range(a.envs)], start_method="spawn"))
    if a.resume:
        vecnorm = VecNormalize.load(str(Path(a.resume) / "last_vecnorm.pkl"), venv)
        model = PPO.load(Path(a.resume) / "last.zip", env=vecnorm, device="cpu")
    else:
        vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.98)
        model = PPO("MlpPolicy", vecnorm, n_steps=512, batch_size=2560, n_epochs=6, learning_rate=3e-4,
                    gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh,
                                       log_std_init=-1.0),
                    device="cpu", verbose=1, tensorboard_log=None, seed=0)
    deadline = time.time() + a.minutes * 60
    cb = Watch(run, vecnorm, deadline, a.eval_every, ckpt_every_s=180)
    t0 = time.time()
    model.learn(total_timesteps=10**9, callback=cb, reset_num_timesteps=not a.resume, log_interval=5)
    model.save(run / "last.zip"); vecnorm.save(str(run / "last_vecnorm.pkl"))
    print(f"DONE {model.num_timesteps} steps in {(time.time() - t0) / 60:.1f} min; best score {cb.best:.3f}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
