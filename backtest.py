"""Evaluate the trained agent on held-out data vs. baselines, each seed, and random "monkeys".

Usage: python backtest.py [--split val|test] [--symbol ETH/USD]
--symbol tests on another coin from train_symbols (default: the coin you trade).
Run on the test split only once you're done tuning - otherwise it stops being "unseen".

Monkey test: 1000 random traders copy the agent's holding periods exactly (same trades, same
time in the market) but at random times, with the same fees. If the agent doesn't beat ~95%
of them, its timing is probably no better than chance.
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
from common import data_path, load_config, load_splits, make_env, metrics, trade_cost, trade_rules, write_json
from train import run_episode


def monkey_returns(close, positions, cost, runs, seed=0):
    """Final returns (%) of `runs` random traders that copy the agent's EXACT holding periods
    (same number of trades, same time in the market, same trade lengths) but place them at
    random times. So the only thing that differs from the agent is the TIMING.
    Same accounting as the env: net_t = log(close[t+1]/close[t]) * pos_t - trades * cost."""
    rng = np.random.default_rng(seed)
    logret = np.log(close[1:] / close[:-1])[:len(positions)]
    pos = np.asarray(positions, dtype=np.int8)
    if len(pos) == 0 or pos.min() == pos.max():        # never traded, or never got out
        return np.full(runs, (np.exp((logret * pos).sum() - (pos[:1].sum() * cost)) - 1) * 100)
    # split into runs of identical position: e.g. flat 40 bars, long 12, flat 7, long 30...
    edges = np.flatnonzero(np.diff(pos)) + 1
    lengths = np.diff(np.concatenate([[0], edges, [len(pos)]]))
    values = pos[np.concatenate([[0], edges])]
    long_runs, flat_runs = lengths[values == 1], lengths[values == 0]
    n_trades = len(edges) + int(pos[0] == 1)            # env starts flat
    out = np.empty(runs)
    for r in range(runs):
        lr, fr = rng.permutation(long_runs), rng.permutation(flat_runs)
        seq, li, fi = [], 0, 0
        for v in values:                                # keep the long/flat alternation
            if v:
                seq.append(np.ones(lr[li], np.int8)); li += 1
            else:
                seq.append(np.zeros(fr[fi], np.int8)); fi += 1
        p = np.concatenate(seq)
        out[r] = (np.exp((logret * p).sum() - n_trades * cost) - 1) * 100
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--symbol", default=None, help="coin to test on (default: config symbol)")
    p.add_argument("--vote", choices=["soft", "majority"], default=None,
                   help="ensemble voting rule (default: ensemble.vote in config.yaml)")
    args = p.parse_args()
    cfg = load_config()
    sym = args.symbol or cfg["symbol"]
    splits = load_splits(cfg)[0]
    if sym not in splits:
        raise SystemExit(f"{sym} has no data - add it to train_symbols and run python -m data.fetch")
    df, feats = splits[sym][args.split]
    tf = cfg["timeframe"]
    cost = trade_cost(cfg)

    agent = load_agent(cfg, vote=args.vote)
    agent_label = agent.label()
    agent_eq, trades, positions = run_episode(agent, make_env(cfg, df, feats))
    start = cfg["train"]["window"] - 1
    close = df["close"].values[start:start + len(agent_eq)]
    bh_eq = close / close[0] * (1 - cfg["fee"] - cfg["slippage"])   # buy once at market, hold
    flat_eq = np.ones_like(agent_eq)               # do nothing

    rows = {"PPO agent": {**metrics(agent_eq, tf), "trades": trades}}
    seeds = load_seed_models(cfg) if len(agent) > 1 else {}
    for name, model in seeds.items():              # each seed on its own, to see the spread
        eq, n_tr, _ = run_episode(model, make_env(cfg, df, feats))
        rows[name] = {**metrics(eq, tf), "trades": n_tr}
    rows["Buy & hold"] = metrics(bh_eq, tf)
    # the 200-day rule on the same candles (the "smaller crashes" benchmark)
    from rules_test import rule_equity_like_bot, trend_rule_at   # here: rules_test imports walkforward
    ts_ms = df["ts"].values[start:start + len(agent_eq)]
    rule_pos = trend_rule_at(pd.read_parquet(data_path(cfg, sym)), ts_ms)
    rows["200-day rule"] = metrics(rule_equity_like_bot(cfg, df.iloc[start:start + len(agent_eq)], rule_pos), tf)
    rows["Do nothing"] = metrics(flat_eq, tf)
    table = pd.DataFrame(rows).T.round(2)

    # --- monkey test ---
    runs = cfg.get("monkey_runs", 1000)
    monkeys = monkey_returns(close, positions, cost, runs)
    agent_ret = (agent_eq[-1] - 1) * 100
    # ties (monkey identical to the agent) count as half, with a small tolerance for rounding
    tol = 1e-6
    beats = float(((agent_ret > monkeys + tol).mean() + 0.5 * (np.abs(agent_ret - monkeys) <= tol).mean()) * 100)
    # with only 1-2 holding periods there is nothing to shuffle: every monkey = the agent
    n_long = int(np.sum(np.diff(np.concatenate([[0], positions])) == 1))
    testable = bool(trades > 0 and monkeys.std() > tol)
    monkey = {"agent_label": agent_label, "symbol": sym, "agent_return_%": round(agent_ret, 3),
              "trades": int(trades), "runs": runs, "beats_%": round(beats, 1),
              "testable": testable, "holding_periods": n_long,
              "monkey_median_%": round(float(np.median(monkeys)), 3),
              "monkey_p5_%": round(float(np.percentile(monkeys, 5)), 3),
              "monkey_p95_%": round(float(np.percentile(monkeys, 95)), 3)}

    tr = trade_rules(cfg)
    print(f"\n{sym} {args.split.upper()} split: {len(agent_eq)} bars of {tf}   "
          f"(row 'PPO agent' = {agent_label})")
    print(f"rules: {tr['order_type']} orders, min_hold {tr['min_hold']}, stop_loss "
          f"{tr['stop_loss'] * 100:.1f}%, cost per trade {cost * 100:.2f}%\n")
    print(table.to_string())
    print(f"\nMonkey test: {runs} random traders with the same {trades} trades and "
          f"{positions.mean() * 100:.0f}% time in market, at random times")
    if trades == 0:
        print("  agent made no trades - nothing to compare (it's the same as 'Do nothing').")
    elif not testable:
        print(f"  can't judge: only {n_long} holding period(s), so there's no timing to shuffle "
              "(every monkey is identical to the agent).")
    else:
        print(f"  agent {agent_ret:+.2f}%  vs monkeys: median {monkey['monkey_median_%']:+.2f}%, "
              f"90% between {monkey['monkey_p5_%']:+.2f}% and {monkey['monkey_p95_%']:+.2f}%")
        verdict = ("likely real skill" if beats >= 95 else
                   "maybe something, not convincing" if beats >= 75 else "no better than chance")
        print(f"  agent beats {beats:.1f}% of monkeys -> {verdict}")
        if n_long < 5:
            print(f"  (only {n_long} holding periods - weak test, take it with a grain of salt)")

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
    write_json(out / f"monkey_{args.split}.json", monkey)

    plt.figure(figsize=(11, 5))
    plt.plot(ts, agent_eq, label=agent_label)
    plt.plot(ts, bh_eq, label="Buy & hold", alpha=0.7)
    plt.plot(ts, flat_eq, label="Do nothing", ls="--", c="gray")
    plt.title(f"{sym} {tf} - {args.split} split (fees + slippage included)")
    plt.ylabel("equity (start = 1.0)")
    plt.legend()
    plt.tight_layout()
    f = out / f"backtest_{args.split}.png"
    plt.savefig(f, dpi=120)
    print(f"\nchart -> {f}")


if __name__ == "__main__":
    main()
