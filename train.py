"""Train PPO agents on the train split, one per random seed; keep the best checkpoint by validation.

Usage:
  python train.py                 # number of seeds from config.yaml (train.n_seeds)
  python train.py --seeds 5       # train 5 agents with different random seeds
  python train.py --timesteps 200000

RL results vary a lot between random seeds - one run can look great (or awful) by luck.
Training several and comparing them tells you how much to trust the result.
The best seed (by validation return) is copied to the main model path, so backtest.py and
paper_trade.py use it automatically.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
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


LOG_FIELDS = ["seed", "timesteps", "elapsed_s", "val_return_%", "val_sharpe", "val_max_drawdown_%",
              "val_trades", "train_ep_reward", "saved"]


class ValCallback(BaseCallback):
    """Evaluates on validation every `every` steps (and at the end), saves the best model, and
    appends a row to the seed's training log (read live by the dashboard)."""

    def __init__(self, val_env, path, log_path, timeframe, seed, every=25_000):
        super().__init__()
        self.val_env, self.path, self.every, self.seed = val_env, path, every, seed
        self.log_path, self.timeframe = Path(log_path), timeframe
        self.best, self.best_row = -np.inf, None
        self.ep_rewards = []
        self.last_eval = -1

    def _on_training_start(self):
        self.t0 = time.time()
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    def _evaluate(self):
        self.last_eval = self.num_timesteps
        eq, trades, _ = run_episode(self.model, self.val_env)
        m = metrics(eq, self.timeframe)
        ret = m["total_return_%"]
        saved = ret > self.best
        ep_r = float(np.mean(self.ep_rewards[-20:])) if self.ep_rewards else float("nan")
        row = [self.seed, self.num_timesteps, round(time.time() - self.t0, 1), round(ret, 3),
               round(m["sharpe"], 3), round(m["max_drawdown_%"], 3), trades, round(ep_r, 3),
               int(saved)]
        if saved:
            self.best, self.best_row = ret, dict(zip(LOG_FIELDS, row))
            self.model.save(self.path)
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  [seed {self.seed}] [{self.num_timesteps:>8}] val return {ret:+.2f}%  "
              f"trades {trades}{'  <- saved' if saved else ''}")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.every < self.training_env.num_envs:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.num_timesteps - self.last_eval >= self.training_env.num_envs:
            self._evaluate()


def train_one(cfg, splits, seed, timesteps, model_path, log_path):
    tr_df, tr_f = splits["train"]
    va_df, va_f = splits["val"]
    venv = make_vec_env(lambda: make_env(cfg, tr_df, tr_f, episode_len=2000), n_envs=4, seed=seed)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=1024, batch_size=256,
                learning_rate=3e-4, ent_coef=0.01, gamma=0.99)
    cb = ValCallback(make_env(cfg, va_df, va_f), model_path, log_path, cfg["timeframe"], seed,
                     every=max(10_000, timesteps // 20))
    model.learn(total_timesteps=timesteps, callback=cb)
    return cb.best_row


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=cfg["train"]["total_timesteps"])
    p.add_argument("--seeds", type=int, default=cfg["train"].get("n_seeds", 1),
                   help="how many agents to train with different random seeds")
    args = p.parse_args()

    splits, norm = load_splits(cfg)
    print(f"bars: train {len(splits['train'][0])}  val {len(splits['val'][0])}  "
          f"test {len(splits['test'][0])}")

    main_path = cfg["paths"]["model"]
    Path(main_path).parent.mkdir(parents=True, exist_ok=True)
    save_norm(cfg, norm)  # paper/live trading must normalize features exactly like training

    report = Path(cfg["paths"]["report"])
    report.mkdir(parents=True, exist_ok=True)
    for old in list(report.glob("training_log*.csv")) + [report / "seeds_summary.csv"]:
        old.unlink(missing_ok=True)  # clear the previous run so the dashboard isn't confused
    seeds = [cfg["train"]["seed"] + i for i in range(args.seeds)]
    (report / "train_run.json").write_text(json.dumps(
        {"seeds": seeds, "timesteps": args.timesteps,
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n=== seed {seed} ({i}/{len(seeds)}) ===")
        best = train_one(cfg, splits, seed, args.timesteps, f"{main_path}_seed{seed}",
                         report / f"training_log_seed{seed}.csv")
        results.append(best)

    summary = pd.DataFrame(results)[["seed", "val_return_%", "val_sharpe", "val_max_drawdown_%",
                                     "val_trades", "timesteps"]]
    summary = summary.rename(columns={"timesteps": "best_at_step"})
    winner = summary.loc[summary["val_return_%"].idxmax()]
    summary["chosen"] = summary["seed"] == winner["seed"]
    summary.to_csv(report / "seeds_summary.csv", index=False)
    shutil.copy(f"{main_path}_seed{int(winner['seed'])}.zip", f"{main_path}.zip")
    # all seeds of this run, for ensemble voting (agents.py)
    Path(f"{main_path}_ensemble.json").write_text(json.dumps(
        {"seeds": seeds, "models": [f"{main_path}_seed{s}" for s in seeds]}, indent=1))

    r = summary["val_return_%"]
    print("\n=== summary (best checkpoint of each seed, on validation) ===")
    print(summary.to_string(index=False))
    if len(summary) > 1:
        print(f"\nval return across seeds: mean {r.mean():+.2f}%  std {r.std():.2f}  "
              f"range {r.min():+.2f}% .. {r.max():+.2f}%")
        if r.std() > max(2.0, abs(r.mean())):
            print("seeds disagree a lot -> results are mostly luck; be skeptical of the best one.")
    print(f"\nbest single seed: {int(winner['seed'])} -> {main_path}.zip")
    if len(seeds) > 1:
        print("backtest.py / paper_trade.py will use the ENSEMBLE of all seeds "
              "(set ensemble.enabled: false in config.yaml to use only the best seed)")


if __name__ == "__main__":
    main()
