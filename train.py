"""Train a PPO agent on the train split; keep the checkpoint with the best validation return.

Usage: python train.py [--timesteps N]
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from common import load_config, load_splits, make_env


def run_episode(model, env):
    obs, _ = env.reset()
    equity, done = [1.0], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        equity.append(info["equity"])
    return np.array(equity), info["n_trades"]


class ValCallback(BaseCallback):
    def __init__(self, val_env, path, every=25_000):
        super().__init__()
        self.val_env, self.path, self.every = val_env, path, every
        self.best = -np.inf

    def _on_step(self):
        if self.num_timesteps % self.every < self.training_env.num_envs:
            eq, trades = run_episode(self.model, self.val_env)
            ret = (eq[-1] - 1) * 100
            tag = ""
            if ret > self.best:
                self.best = ret
                self.model.save(self.path)
                tag = "  <- saved"
            print(f"[{self.num_timesteps:>8}] val return {ret:+.2f}%  trades {trades}{tag}")
        return True


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=cfg["train"]["total_timesteps"])
    args = p.parse_args()

    splits = load_splits(cfg)
    tr_df, tr_f = splits["train"]
    va_df, va_f = splits["val"]
    print(f"bars: train {len(tr_df)}  val {len(va_df)}  test {len(splits['test'][0])}")

    seed = cfg["train"]["seed"]
    venv = make_vec_env(lambda: make_env(cfg, tr_df, tr_f, episode_len=2000), n_envs=4, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=1024, batch_size=256,
                learning_rate=3e-4, ent_coef=0.01, gamma=0.99)

    path = cfg["paths"]["model"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cb = ValCallback(make_env(cfg, va_df, va_f), path, every=max(10_000, args.timesteps // 20))
    model.learn(total_timesteps=args.timesteps, callback=cb)
    if not Path(path + ".zip").exists():
        model.save(path)
    print(f"best val return {cb.best:+.2f}%  model -> {path}.zip")


if __name__ == "__main__":
    main()
