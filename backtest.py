"""Evaluate the trained agent on held-out data vs. baselines, each seed, and random "monkeys".

Usage: python backtest.py [--split val|test]
Run on the test split only once you're done tuning - otherwise it stops being "unseen".

Monkey test: 1000 random traders make the SAME number of trades as the agent, at random
times, with the same fees. If the agent doesn't beat ~95% of them, its timing is
probably no better than chance.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import load_agent, load_seed_models
from common import load_config, load_splits, make_env, metrics
from train import run_episode


def monkey_returns(close, n_trades, cost, runs, seed=0):
    """Final returns (%) of `runs` random long/flat traders with exactly `n_trades` position
    changes at random bars, starting flat. Same accounting as the env:
    net_t = log(close[t+1]/close[t]) * pos_t - |pos_t - pos_{t-1}| * cost."""
    rng = np.random.default_rng(seed)
    logret = np.log(close[1:] / close[:-1])          # one per decision bar
    n = len(logret)
    k = min(n_trades, n)
    if k == 0:
        return np.zeros(runs)
    out = np.empty(runs)
    for r in range(runs):
        flips = np.zeros(n, dtype=np.int8)
        flips[rng.choice(n, size=k, replace=False)] = 1
        pos = np.cumsum(flips) % 2                   # each flip toggles flat <-> long
        out[r] = (np.exp((logret * pos).sum() - k * cost) - 1) * 100
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()
    cfg = load_config()
    df, feats = load_splits(cfg)[0][args.split]
    tf = cfg["timeframe"]
    cost = cfg["fee"] + cfg["slippage"]

    agent = load_agent(cfg)
    agent_label = f"Ensemble ({len(agent)} seeds)" if len(agent) > 1 else "PPO agent"
    agent_eq, trades, positions = run_episode(agent, make_env(cfg, df, feats))
    start = cfg["train"]["window"] - 1
    close = df["close"].values[start:start + len(agent_eq)]
    bh_eq = close / close[0] * (1 - cost)          # buy once, hold
    flat_eq = np.ones_like(agent_eq)               # do nothing

    rows = {"PPO agent": {**metrics(agent_eq, tf), "trades": trades}}
    seeds = load_seed_models(cfg) if len(agent) > 1 else {}
    for name, model in seeds.items():              # each seed on its own, to see the spread
        eq, n_tr, _ = run_episode(model, make_env(cfg, df, feats))
        rows[name] = {**metrics(eq, tf), "trades": n_tr}
    rows["Buy & hold"] = metrics(bh_eq, tf)
    rows["Do nothing"] = metrics(flat_eq, tf)
    table = pd.DataFrame(rows).T.round(2)

    # --- monkey test ---
    runs = cfg.get("monkey_runs", 1000)
    monkeys = monkey_returns(close, trades, cost, runs)
    agent_ret = (agent_eq[-1] - 1) * 100
    beats = float((agent_ret > monkeys).mean() * 100)
    monkey = {"agent_label": agent_label, "agent_return_%": round(agent_ret, 3),
              "trades": int(trades), "runs": runs, "beats_%": round(beats, 1),
              "monkey_median_%": round(float(np.median(monkeys)), 3),
              "monkey_p5_%": round(float(np.percentile(monkeys, 5)), 3),
              "monkey_p95_%": round(float(np.percentile(monkeys, 95)), 3)}

    print(f"\n{args.split.upper()} split: {len(agent_eq)} bars of {tf}   "
          f"(row 'PPO agent' = {agent_label})\n")
    print(table.to_string())
    print(f"\nMonkey test: {runs} random traders with the same {trades} trades")
    if trades == 0:
        print("  agent made no trades - nothing to compare (it's the same as 'Do nothing').")
    else:
        print(f"  agent {agent_ret:+.2f}%  vs monkeys: median {monkey['monkey_median_%']:+.2f}%, "
              f"90% between {monkey['monkey_p5_%']:+.2f}% and {monkey['monkey_p95_%']:+.2f}%")
        verdict = ("likely real skill" if beats >= 95 else
                   "maybe something, not convincing" if beats >= 75 else "no better than chance")
        print(f"  agent beats {beats:.1f}% of monkeys -> {verdict}")

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
    pd.DataFrame({"return_%": monkeys.round(3)}).to_csv(out / f"monkeys_{args.split}.csv", index=False)
    (out / f"monkey_{args.split}.json").write_text(json.dumps(monkey, indent=1))

    plt.figure(figsize=(11, 5))
    plt.plot(ts, agent_eq, label=agent_label)
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
