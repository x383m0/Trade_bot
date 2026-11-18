"""Train a PPO agent on the train split; keep the checkpoint with the best validation return.

Usage: python train.py [--timesteps N]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from common import load_config, load_splits, make_env, metrics, save_norm


def run_episode(model, env):
    """Returns (equity curve, n_trades, positions). positions[i] = position held from bar i to i+1."""
    obs, _ = env.reset()
    equity, positions, done = [1.0], [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        equity.append(info["equity"])
        positions.append(int(action))
    return np.array(equity), info["n_trades"], np.array(positions)


LOG_FIELDS = ["timesteps", "elapsed_s", "val_return_%", "val_sharpe", "val_max_drawdown_%",
              "val_trades", "train_ep_reward", "saved"]


class ValCallback(BaseCallback):
    """Evaluates on validation every `every` steps, saves the best model, and appends a row
    to reports/training_log.csv (read live by the dashboard)."""

    def __init__(self, val_env, path, log_path, timeframe, every=25_000):
        super().__init__()
        self.val_env, self.path, self.every = val_env, path, every
        self.log_path, self.timeframe = Path(log_path), timeframe
        self.best = -np.inf
        self.ep_rewards = []

    def _on_training_start(self):
        self.t0 = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.every < self.training_env.num_envs:
            eq, trades, _ = run_episode(self.model, self.val_env)
            m = metrics(eq, self.timeframe)
            ret = m["total_return_%"]
            saved = ret > self.best
            if saved:
                self.best = ret
                self.model.save(self.path)
            ep_r = float(np.mean(self.ep_rewards[-20:])) if self.ep_rewards else float("nan")
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, round(time.time() - self.t0, 1),
                                        round(ret, 3), round(m["sharpe"], 3),
                                        round(m["max_drawdown_%"], 3), trades,
                                        round(ep_r, 3), int(saved)])
            print(f"[{self.num_timesteps:>8}] val return {ret:+.2f}%  trades {trades}"
                  f"{'  <- saved' if saved else ''}")
        return True


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=cfg["train"]["total_timesteps"])
    args = p.parse_args()

    splits, norm = load_splits(cfg)
    tr_df, tr_f = splits["train"]
    va_df, va_f = splits["val"]
    print(f"bars: train {len(tr_df)}  val {len(va_df)}  test {len(splits['test'][0])}")

    seed = cfg["train"]["seed"]
    venv = make_vec_env(lambda: make_env(cfg, tr_df, tr_f, episode_len=2000), n_envs=4, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=1024, batch_size=256,
                learning_rate=3e-4, ent_coef=0.01, gamma=0.99)

    path = cfg["paths"]["model"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_norm(cfg, norm)  # paper/live trading must normalize features exactly like training
    log_path = Path(cfg["paths"]["report"]) / "training_log.csv"
    cb = ValCallback(make_env(cfg, va_df, va_f), path, log_path, cfg["timeframe"],
                     every=max(10_000, args.timesteps // 20))
    model.learn(total_timesteps=args.timesteps, callback=cb)
    if not Path(path + ".zip").exists():
        model.save(path)
    print(f"best val return {cb.best:+.2f}%  model -> {path}.zip")


if __name__ == "__main__":
    main()
