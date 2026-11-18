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
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from common import load_config, load_splits, make_env, metrics, pair_of, save_norm


def run_episode(model, env):
    """Returns (equity curve, n_trades, positions). positions[i] = position held from bar i to i+1."""
    obs, _ = env.reset()
    equity, positions, done = [1.0], [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        equity.append(info["equity"])
        positions.append(int(info["pos"]))   # actual position (rules/unfilled orders can override)
    return np.array(equity), info["n_trades"], np.array(positions)


BASE_FIELDS = ["seed", "timesteps", "elapsed_s", "val_return_%", "val_sharpe", "val_max_drawdown_%",
               "val_trades", "train_ep_reward", "saved"]


class ValCallback(BaseCallback):
    """Evaluates on every coin's validation period every `every` steps (and at the end), saves
    the best model (by AVERAGE validation return across coins - more data = less luck), and
    appends a row to the seed's training log (read live by the dashboard)."""

    def __init__(self, val_envs, path, log_path, timeframe, seed, every=25_000):
        super().__init__()
        self.val_envs, self.path, self.every, self.seed = val_envs, path, every, seed
        self.log_path, self.timeframe = Path(log_path), timeframe
        self.fields = BASE_FIELDS + [f"val_ret_{pair_of(s)}" for s in val_envs]
        self.best, self.best_row = -np.inf, None
        self.ep_rewards = []
        self.last_eval = -1

    def _on_training_start(self):
        self.t0 = time.time()
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.fields)

    def _evaluate(self):
        self.last_eval = self.num_timesteps
        per = {}
        for sym, env in self.val_envs.items():
            eq, trades, _ = run_episode(self.model, env)
            per[sym] = {**metrics(eq, self.timeframe), "trades": trades}
        avg = lambda k: float(np.mean([m[k] for m in per.values()]))
        ret = avg("total_return_%")
        saved = ret > self.best
        ep_r = float(np.mean(self.ep_rewards[-20:])) if self.ep_rewards else float("nan")
        row = [self.seed, self.num_timesteps, round(time.time() - self.t0, 1), round(ret, 3),
               round(avg("sharpe"), 3), round(avg("max_drawdown_%"), 3),
               int(round(avg("trades"))), round(ep_r, 3), int(saved)]
        row += [round(m["total_return_%"], 3) for m in per.values()]
        if saved:
            self.best, self.best_row = ret, dict(zip(self.fields, row))
            self.model.save(self.path)
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        coins = "  ".join(f"{pair_of(s)} {m['total_return_%']:+.1f}%" for s, m in per.items())
        print(f"  [seed {self.seed}] [{self.num_timesteps:>8}] val return avg {ret:+.2f}%  "
              f"({coins})  trades {int(round(avg('trades')))}{'  <- saved' if saved else ''}")

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


def torch_threads(cfg, workers=1):
    """CPU threads per training process. This network is tiny, so splitting each calculation
    over many threads costs more than it saves: 2 is fastest (measured), more is SLOWER."""
    import torch
    want = cfg["train"].get("threads") or 2
    n = max(1, min(int(want), (os.cpu_count() or 2) // max(1, workers)))
    torch.set_num_threads(n)
    return n


def train_one(cfg, splits, seed, timesteps, model_path, log_path):
    syms = list(splits)
    tr_df = [splits[s]["train"][0] for s in syms]
    tr_f = [splits[s]["train"][1] for s in syms]
    venv = make_vec_env(lambda: make_env(cfg, tr_df, tr_f, episode_len=2000), n_envs=4, seed=seed)
    t = cfg["train"]
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=1024, batch_size=256,
                learning_rate=t.get("learning_rate", 3e-4), ent_coef=t.get("ent_coef", 0.01),
                gamma=t.get("gamma", 0.99), device=t.get("device", "cpu"))
    val_envs = {s: make_env(cfg, *splits[s]["val"]) for s in syms}
    unlimited = not timesteps
    cb = ValCallback(val_envs, model_path, log_path, cfg["timeframe"], seed,
                     every=25_000 if unlimited else max(10_000, timesteps // 20))
    try:
        model.learn(total_timesteps=UNLIMITED if unlimited else timesteps, callback=cb)
    except KeyboardInterrupt:
        # Ctrl+C: stop this seed but keep its best checkpoint (already saved by the callback)
        global INTERRUPTED
        INTERRUPTED = True
        print(f"\n  [seed {seed}] stopped at {model.num_timesteps:,} steps - keeping its best checkpoint")
    return cb.best_row


UNLIMITED = 10 ** 12   # "--timesteps 0": train until Ctrl+C
INTERRUPTED = False


def _train_seed_worker(seed, timesteps, model_path, log_path, workers, fractions=None):
    """Runs in its own process (--parallel): loads the data itself and trains one seed,
    with a small number of CPU threads so parallel seeds don't fight over the cores."""
    cfg = load_config()
    torch_threads(cfg, workers)
    splits, _ = load_splits(cfg, fractions)
    return train_one(cfg, splits, seed, timesteps, model_path, log_path)


def train_seeds(cfg, splits, seeds, timesteps, model_prefix, log_dir, parallel=1, fractions=None):
    """Train several seeds (one after another, or `parallel` at a time). Each seed's best
    checkpoint is saved to {model_prefix}_seed{N}.zip and its log to log_dir.
    Returns (best rows of the seeds that saved a checkpoint, stopped_with_ctrl_c)."""
    log_dir = Path(log_dir)
    workers = max(1, min(parallel, len(seeds)))
    if not timesteps and len(seeds) > workers:
        workers = len(seeds)       # unlimited one-after-another would never reach seed 2
    stopped = False
    if workers == 1:
        print(f"(using {torch_threads(cfg)} CPU threads - more is slower for this small network; "
              f"use --parallel to train several seeds at once)")
        results = []
        for i, seed in enumerate(seeds, 1):
            print(f"\n=== seed {seed} ({i}/{len(seeds)}) ===")
            results.append(train_one(cfg, splits, seed, timesteps, f"{model_prefix}_seed{seed}",
                                     log_dir / f"training_log_seed{seed}.csv"))
            if INTERRUPTED:
                stopped = True
                print("Ctrl+C: skipping the remaining seeds")
                break
    else:
        cores = os.cpu_count() or 2
        per = max(1, min(int(cfg["train"].get("threads") or 2), cores // workers))
        print(f"\n=== training {len(seeds)} seeds, {workers} at a time "
              f"({per} threads each, {cores} CPU threads available) ===")
        # "spawn" works the same on Windows, Linux and Mac
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(_train_seed_worker, seed, timesteps, f"{model_prefix}_seed{seed}",
                                 str(log_dir / f"training_log_seed{seed}.csv"), workers, fractions)
                       for seed in seeds]
            try:
                for f in futures:
                    f.result()
            except KeyboardInterrupt:
                stopped = True
                print("\nCtrl+C: stopping all seeds and keeping their best checkpoints...")
            results = []
            for f in futures:
                try:
                    results.append(f.result())
                except BaseException:   # a seed that was stopped before finishing setup
                    results.append(None)
    return [r for r in results if r], stopped


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=cfg["train"]["total_timesteps"],
                   help="training steps per seed; 0 = keep training until you press Ctrl+C")
    p.add_argument("--seeds", type=int, default=cfg["train"].get("n_seeds", 1),
                   help="how many agents to train with different random seeds")
    p.add_argument("--parallel", type=int, default=cfg["train"].get("parallel", 1),
                   help="train this many seeds at the same time (use on CPUs with many cores)")
    args = p.parse_args()

    splits, norm = load_splits(cfg)
    for sym, sp in splits.items():
        print(f"{sym:>9}: train {len(sp['train'][0]):>6} bars  val {len(sp['val'][0]):>6}  "
              f"test {len(sp['test'][0]):>6}")
    print(f"learning from {len(splits)} coin(s), trading {cfg['symbol']}")

    main_path = cfg["paths"]["model"]
    Path(main_path).parent.mkdir(parents=True, exist_ok=True)
    save_norm(cfg, norm)  # paper/live trading must normalize features exactly like training

    report = Path(cfg["paths"]["report"])
    report.mkdir(parents=True, exist_ok=True)
    for old in list(report.glob("training_log*.csv")) + [report / "seeds_summary.csv"]:
        old.unlink(missing_ok=True)  # clear the previous run so the dashboard isn't confused
    seeds = [cfg["train"]["seed"] + i for i in range(args.seeds)]
    (report / "train_run.json").write_text(json.dumps(
        {"seeds": seeds, "timesteps": args.timesteps or None, "coins": list(splits),
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    if not args.timesteps:
        print(f"\nUNLIMITED training: press Ctrl+C when you want to stop. The best checkpoint "
              f"of each seed is kept, then the summary and ensemble are written as usual.")
    results, _ = train_seeds(cfg, splits, seeds, args.timesteps, main_path, report, args.parallel)
    if not results:
        raise SystemExit("No checkpoint was saved yet (stopped before the first validation check).")
    seeds = [int(r["seed"]) for r in results]

    coin_cols = [c for c in results[0] if c.startswith("val_ret_")]
    summary = pd.DataFrame(results)[["seed", "val_return_%", "val_sharpe", "val_max_drawdown_%",
                                     "val_trades", "timesteps"] + coin_cols]
    summary = summary.rename(columns={"timesteps": "best_at_step"})
    winner = summary.loc[summary["val_return_%"].idxmax()]
    summary["chosen"] = summary["seed"] == winner["seed"]
    summary.to_csv(report / "seeds_summary.csv", index=False)
    shutil.copy(f"{main_path}_seed{int(winner['seed'])}.zip", f"{main_path}.zip")
    # all seeds of this run, for ensemble voting (agents.py)
    Path(f"{main_path}_ensemble.json").write_text(json.dumps(
        {"seeds": seeds, "models": [f"{main_path}_seed{s}" for s in seeds]}, indent=1))

    r = summary["val_return_%"]
    print("\n=== summary (best checkpoint of each seed; val_return_% = average over coins) ===")
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
