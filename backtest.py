"""Evaluate the trained agent on the held-out TEST split vs. baselines.

Usage: python backtest.py [--split test|val]
Run on the test split only once you're done tuning - otherwise it stops being "unseen".
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from common import load_config, load_splits, make_env, metrics
from train import run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()
    cfg = load_config()
    df, feats = load_splits(cfg)[0][args.split]
    env = make_env(cfg, df, feats)
    model = PPO.load(cfg["paths"]["model"])

    agent_eq, trades, positions = run_episode(model, env)
    start = cfg["train"]["window"] - 1
    close = df["close"].values[start:start + len(agent_eq)]
    cost = cfg["fee"] + cfg["slippage"]
    bh_eq = close / close[0] * (1 - cost)          # buy once, hold
    flat_eq = np.ones_like(agent_eq)               # do nothing

    tf = cfg["timeframe"]
    rows = {"PPO agent": metrics(agent_eq, tf), "Buy & hold": metrics(bh_eq, tf),
            "Do nothing": metrics(flat_eq, tf)}
    table = pd.DataFrame(rows).T.round(2)
    table.loc["PPO agent", "trades"] = trades
    print(f"\n{args.split.upper()} split: {len(agent_eq)} bars of {tf}\n")
    print(table.to_string())

    out = Path(cfg["paths"]["report"])
    out.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(df["ts"].values[start:start + len(agent_eq)], unit="ms")

    # data for the dashboard
    ohlc = df.iloc[start:start + len(agent_eq)]
    pd.DataFrame({"time": ts, "open": ohlc["open"].values, "high": ohlc["high"].values,
                  "low": ohlc["low"].values, "close": close, "agent": agent_eq, "buy_hold": bh_eq,
                  "position": np.append(positions, positions[-1] if len(positions) else 0)}
                 ).to_csv(out / f"equity_{args.split}.csv", index=False)
    table.to_csv(out / f"metrics_{args.split}.csv")
    plt.figure(figsize=(11, 5))
    plt.plot(ts, agent_eq, label="PPO agent")
    plt.plot(ts, bh_eq, label="Buy & hold", alpha=0.7)
    plt.plot(ts, flat_eq, label="Do nothing", ls="--", c="gray")
    plt.title(f"{cfg['symbol']} {tf} - {args.split} split (fees + slippage included)")
    plt.ylabel("equity (start = 1.0)")
    plt.legend()
    plt.tight_layout()
    f = out / f"backtest_{args.split}.png"
    plt.savefig(f, dpi=120)
    print(f"\nchart -> {f}")


if __name__ == "__main__":
    main()
