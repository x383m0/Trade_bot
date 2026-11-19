"""Walk-forward test: is the bot good in MANY periods, or did one backtest get lucky?

Usage:
  python walkforward.py                       # settings from config.yaml -> walkforward
  python walkforward.py --folds 6 --timesteps 300000 --seeds 3 --parallel 3
  python walkforward.py --fresh               # redo every fold (otherwise finished folds are reused)

How it works (with folds 6, start 0.40, end 0.85):

  |------ train ------|-- val --|-- TEST 1 --|
  |--------- train --------|-- val --|-- TEST 2 --|
  ...                                        ... |-- TEST 6 --|  [last 15% untouched]

For each fold a fresh set of seeds is trained ONLY on data before its test window
(the chunk right before the test window picks the best checkpoints), then the ensemble trades
the test window. Nothing from a test window is ever used to train or choose its own bot.
The last part of history (after `end`, i.e. the normal --split test) is never touched, so it
stays a true final exam.

Result: how many periods the bot beat buy & hold, by how much, and the stitched-together equity.
Finished folds are saved, so you can stop with Ctrl+C and rerun to continue.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import build_agent
from backtest import monkey_returns
from common import data_path, load_config, load_splits, make_env, metrics, pair_of, read_json, trade_cost, write_json
from train import run_episode, train_seeds


def settings_signature(cfg, folds, start, end, timesteps, seeds):
    """Fingerprint of everything that affects a fold's result (to know when to redo folds)."""
    keys = {k: cfg.get(k) for k in ("symbol", "train_symbols", "timeframe", "years", "fee",
                                    "slippage", "trade_penalty", "orders", "rules", "reward")}
    keys.update(train={k: v for k, v in cfg["train"].items() if k not in ("parallel",)},
                vote=(cfg.get("ensemble") or {}), wf=[folds, start, end, timesteps, seeds])
    return hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]


def fold_windows(folds, start, end):
    """(val_start, test_start, test_end) fractions of history for each fold."""
    length = (end - start) / folds
    return [(round(start + i * length - length, 6), round(start + i * length, 6),
             round(start + (i + 1) * length, 6)) for i in range(folds)]


def run_fold(cfg, i, fracs, seeds, timesteps, parallel, out_dir, model_dir):
    fold_dir = out_dir / f"fold{i}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    splits, _ = load_splits(cfg, fracs)
    sym = cfg["symbol"]
    tr, va, te = (splits[sym][k][0] for k in ("train", "val", "test"))
    fmt = lambda d: pd.to_datetime(d["ts"].iloc[[0, -1]], unit="ms").dt.strftime("%Y-%m-%d").tolist()
    print(f"\n######## fold {i}: train until {fmt(tr)[1]}, choose on {fmt(va)[0]}..{fmt(va)[1]}, "
          f"TEST {fmt(te)[0]}..{fmt(te)[1]} ########")
    prefix = str(model_dir / f"fold{i}")
    results, stopped = train_seeds(cfg, splits, seeds, timesteps, prefix, fold_dir, parallel, fracs)
    if stopped or not results:
        return None
    seed_ids = [int(r["seed"]) for r in results]

    # --- trade the unseen test window ---
    df, feats = splits[sym]["test"]
    agent = build_agent(cfg, [f"{prefix}_seed{s}" for s in seed_ids], [f"seed {s}" for s in seed_ids])
    eq, trades, positions = run_episode(agent, make_env(cfg, df, feats))
    tf, start = cfg["timeframe"], cfg["train"]["window"] - 1
    close = df["close"].values[start:start + len(eq)]
    bh = close / close[0] * (1 - cfg["fee"] - cfg["slippage"])
    m_bot, m_bh = metrics(eq, tf), metrics(bh, tf)
    from rules_test import rule_equity_like_bot, trend_rule_at      # here: rules_test imports this file
    # the 200-day rule on the same candles: the "smaller crashes" benchmark the bot should at least match
    ts_ms = df["ts"].values[start:start + len(eq)]
    rule_pos = trend_rule_at(pd.read_parquet(data_path(cfg, sym)), ts_ms)
    rule = rule_equity_like_bot(cfg, df.iloc[start:start + len(eq)], rule_pos)   # same order rules as the bot
    m_rule = metrics(rule, tf)
    # the same window again, but starting with BTC ALREADY BOUGHT: does it get out when the trend turns?
    eq_l, trades_l, pos_l = run_episode(agent, make_env(cfg, df, feats, start_long=1.0))
    m_long = metrics(eq_l, tf)
    out = rule_pos[:len(pos_l)] == 0                    # candles where the 200-day rule says "cash"
    stuck = round(float(pos_l[out].mean() * 100), 1) if out.any() else None
    per_seed = {}
    for s in seed_ids:
        e, n, _ = run_episode(build_agent(cfg, [f"{prefix}_seed{s}"]), make_env(cfg, df, feats))
        per_seed[f"seed {s}"] = round((e[-1] - 1) * 100, 2)
    monkeys = monkey_returns(close, positions, trade_cost(cfg), cfg.get("monkey_runs", 1000))
    bot_ret = (eq[-1] - 1) * 100
    testable = bool(trades > 0 and monkeys.std() > 1e-6)
    beats = float(((bot_ret > monkeys + 1e-6).mean() + 0.5 * (np.abs(bot_ret - monkeys) <= 1e-6).mean()) * 100)
    ts = pd.to_datetime(df["ts"].values[start:start + len(eq)], unit="ms")
    pd.DataFrame({"time": ts, "bot": eq, "bot_long": eq_l, "buy_hold": bh, "rule_200d": rule}).to_csv(
        fold_dir / "equity.csv", index=False)
    res = {"fold": i, "test_start": str(ts[0].date()), "test_end": str(ts[-1].date()),
           "bot_%": round(bot_ret, 2), "buy_hold_%": round((bh[-1] - 1) * 100, 2),
           "excess_%": round(bot_ret - (bh[-1] - 1) * 100, 2),
           "bot_max_dd_%": round(m_bot["max_drawdown_%"], 2), "bh_max_dd_%": round(m_bh["max_drawdown_%"], 2),
           "rule_%": round((rule[-1] - 1) * 100, 2), "rule_max_dd_%": round(m_rule["max_drawdown_%"], 2),
           "bot_long_%": round((eq_l[-1] - 1) * 100, 2), "bot_long_max_dd_%": round(m_long["max_drawdown_%"], 2),
           "trades_long": int(trades_l), "held_in_downtrend_%": stuck,
           "bot_sharpe": round(m_bot["sharpe"], 2), "trades": int(trades),
           "in_market_%": round(float(positions.mean() * 100), 1),
           "monkey_beats_%": round(beats, 1) if testable else None, "seeds": per_seed,
           "agent": agent.label() if hasattr(agent, "label") else "agent"}
    print(f"  fold {i}: bot {res['bot_%']:+.1f}% (starting with BTC {res['bot_long_%']:+.1f}%)  "
          f"vs buy & hold {res['buy_hold_%']:+.1f}%  vs 200-day rule {res['rule_%']:+.1f}%  "
          f"({res['trades']} / {res['trades_long']} trades"
          + (f", held BTC {stuck:.0f}% of the time the rule said cash)" if stuck is not None else ")"))
    return res


def summarize(cfg, results, out_dir):
    df = pd.DataFrame(results).sort_values("fold")
    won = int((df["bot_%"] > df["buy_hold_%"]).sum())
    positive = int((df["bot_%"] > 0).sum())
    bot_total = (np.prod(1 + df["bot_%"] / 100) - 1) * 100
    bh_total = (np.prod(1 + df["buy_hold_%"] / 100) - 1) * 100
    summ = {"folds": len(df), "beat_buy_hold": won, "positive": positive,
            "avg_excess_%": round(float(df["excess_%"].mean()), 2),
            "bot_total_%": round(float(bot_total), 2), "buy_hold_total_%": round(float(bh_total), 2),
            "worst_bot_dd_%": round(float(df["bot_max_dd_%"].min()), 2),
            "worst_bh_dd_%": round(float(df["bh_max_dd_%"].min()), 2),
            "median_monkey_%": (round(float(df["monkey_beats_%"].dropna().median()), 1)
                                if df["monkey_beats_%"].notna().any() else None)}
    has_rule = "rule_%" in df and df["rule_%"].notna().all()     # folds from before this was added lack it
    has_long = "bot_long_%" in df and df["bot_long_%"].notna().all()
    cols = ["fold", "test_start", "test_end", "bot_%"] + (["bot_long_%"] if has_long else []) + ["buy_hold_%"]
    cols += (["rule_%"] if has_rule else []) + ["trades"] + (["trades_long", "held_in_downtrend_%"] if has_long else [])
    cols += ["bot_max_dd_%"] + (["bot_long_max_dd_%"] if has_long else []) + ["bh_max_dd_%"]
    cols += (["rule_max_dd_%"] if has_rule else []) + ["monkey_beats_%"]
    show = df[cols]
    show.to_csv(out_dir / "summary.csv", index=False)
    write_json(out_dir / "summary.json", summ)
    print(f"\n================ WALK-FORWARD RESULT ({cfg['symbol']} {cfg['timeframe']}) ================")
    print(show.to_string(index=False))
    print(f"\nbeat buy & hold in {won}/{len(df)} periods, made money in {positive}/{len(df)}")
    print(f"average difference vs buy & hold per period: {summ['avg_excess_%']:+.1f}%")
    print(f"all periods chained: bot {bot_total:+.1f}%  vs  buy & hold {bh_total:+.1f}%")
    print(f"worst drawdown in a period: bot {summ['worst_bot_dd_%']:.1f}%  vs  buy & hold "
          f"{summ['worst_bh_dd_%']:.1f}%")
    if has_long:
        long_total = (np.prod(1 + df["bot_long_%"] / 100) - 1) * 100
        stuck = df["held_in_downtrend_%"].dropna()
        summ.update({"bot_long_total_%": round(float(long_total), 2),
                     "worst_bot_long_dd_%": round(float(df["bot_long_max_dd_%"].min()), 2),
                     "held_in_downtrend_%": round(float(stuck.mean()), 1) if len(stuck) else None})
        write_json(out_dir / "summary.json", summ)
        print(f"\nstarting each period WITH BTC already bought: chained {long_total:+.1f}%, worst drop "
              f"{summ['worst_bot_long_dd_%']:.1f}%")
        if len(stuck):
            print(f"held BTC {stuck.mean():.0f}% of the time the 200-day rule said 'cash' (0% = always got out, "
                  "100% = never sold)")
        # judge on the WORSE of the two starts: a bot that only knows when to buy can't look good
        bot_total = min(bot_total, long_total)
        summ["worst_bot_dd_%"] = min(summ["worst_bot_dd_%"], summ["worst_bot_long_dd_%"])
    if has_rule:
        rule_total = (np.prod(1 + df["rule_%"] / 100) - 1) * 100
        beat_rule = int((df["bot_%"] > df["rule_%"] + 0.5).sum())
        summ.update({"rule_total_%": round(float(rule_total), 2), "beat_rule": beat_rule,
                     "worst_rule_dd_%": round(float(df["rule_max_dd_%"].min()), 2)})
        write_json(out_dir / "summary.json", summ)
        print(f"\n200-day rule (the 'smaller crashes' benchmark): chained {rule_total:+.1f}%, worst drop "
              f"{summ['worst_rule_dd_%']:.1f}%. Bot beat it in {beat_rule}/{len(df)} periods"
              + (" (starting in cash)" if has_long else ""))
        if has_long:
            print(f"judged on the worse start: bot {bot_total:+.1f}%, worst drop {summ['worst_bot_dd_%']:.1f}%")
        if bot_total >= rule_total and summ["worst_bot_dd_%"] >= summ["worst_rule_dd_%"] - 5:
            print("vs 200-day rule: ✅ the bot made at least as much with crashes no deeper -> it learned "
                  "something real")
        elif summ["worst_bot_dd_%"] > summ["worst_bh_dd_%"] + 10 and bot_total > 0:
            print("vs 200-day rule: 🤔 the bot cut crashes vs buy & hold but made less than the simple rule")
        else:
            print("vs 200-day rule: ❌ the simple rule did better - the bot hasn't learned the long-term trend")
    share = won / len(df)
    verdict = ("✅ consistently beats buy & hold - promising (the final exam is used up: paper trade it live)"
               if share >= 0.7 and summ["avg_excess_%"] > 0 else
               "🤔 mixed: wins some periods, loses others" if share >= 0.4 else
               "❌ mostly loses to buy & hold")
    print(f"verdict: {verdict}")


def main():
    cfg = load_config()
    wf = cfg.get("walkforward") or {}
    p = argparse.ArgumentParser(description="Walk-forward test over several periods")
    p.add_argument("--folds", type=int, default=wf.get("folds", 6))
    p.add_argument("--timesteps", type=int, default=wf.get("timesteps", 300000))
    p.add_argument("--seeds", type=int, default=cfg["train"].get("n_seeds", 3))
    p.add_argument("--parallel", type=int, default=cfg["train"].get("parallel", 1))
    p.add_argument("--fresh", action="store_true", help="redo all folds, even finished ones")
    args = p.parse_args()
    if not args.timesteps:
        raise SystemExit("walk-forward needs a fixed --timesteps (unlimited isn't possible here)")
    start, end = wf.get("start", 0.40), wf.get("end", 0.85)
    windows = fold_windows(args.folds, start, end)
    if windows[0][0] < 0.15:
        raise SystemExit("first fold would have too little training data - raise walkforward.start "
                         "or use fewer folds")

    out_dir = Path(cfg["paths"]["report"]) / "walkforward"
    model_dir = Path("models") / f"wf_{pair_of(cfg['symbol'])}_{cfg['timeframe']}"
    model_dir.mkdir(parents=True, exist_ok=True)
    sig = settings_signature(cfg, args.folds, start, end, args.timesteps, args.seeds)
    old = read_json(out_dir / "run.json")
    if old and old.get("signature") != sig and (out_dir / "summary.csv").exists():
        # different settings: keep the previous run's results instead of overwriting them
        keep = out_dir.with_name(f"walkforward_{old['signature']}")
        if not keep.exists():
            out_dir.rename(keep)
            print(f"settings changed - the previous walk-forward results were kept in {keep}")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [cfg["train"]["seed"] + i for i in range(args.seeds)]
    print(f"walk-forward: {args.folds} periods between {start:.0%} and {end:.0%} of history, "
          f"{args.seeds} seeds x {args.timesteps:,} steps per period "
          f"(total {args.folds * args.seeds * args.timesteps:,} steps). Ctrl+C to pause.")
    write_json(out_dir / "run.json", {"folds": args.folds, "signature": sig,
                                      "timesteps": args.timesteps, "seeds": args.seeds})

    results = []
    for i, fracs in enumerate(windows, 1):
        f = out_dir / f"fold{i}" / "result.json"
        if f.exists() and not args.fresh:
            r = json.loads(f.read_text())
            if r.get("signature") == sig:
                print(f"fold {i}: already done (bot {r['bot_%']:+.1f}% vs buy & hold "
                      f"{r['buy_hold_%']:+.1f}%) - reusing. Use --fresh to redo.")
                results.append(r)
                continue
        r = run_fold(cfg, i, fracs, seeds, args.timesteps, args.parallel, out_dir, model_dir)
        if r is None:
            print(f"\nstopped during fold {i}. Run the same command again to continue from fold {i}.")
            break
        r["signature"] = sig
        write_json(f, r)
        results.append(r)
    if results:
        summarize(cfg, results, out_dir)


if __name__ == "__main__":
    main()
