"""Coin rotation: hold whichever coin is strongest right now (or cash), on the walk-forward periods.

Usage:
  python rotation_test.py

The idea: instead of "BTC or cash", ask "which of BTC / ETH / SOL went up most lately?" and hold
that one. Coins take turns leading, so following the leader ("momentum") has a better track record
than timing a single coin. With the cash filter, it goes to cash when even the best coin is down.

Decisions are made at the daily close (00:00 UTC), using only closes up to that day. A switch
from one coin to another costs a sell + a buy (2 x the cost per trade in config.yaml).
Graded on exactly the same 6 periods as walkforward.py / rules_test.py, against:
  - buy & hold BTC (what everything has been compared with so far)
  - buy & hold all coins equally (does rotating beat just owning all of them?)

Several variants are tested (lookback, how often it switches). Look for CONSISTENCY across
variants: if only one combination wins, that's probably luck - with enough variants, one always
looks good by chance.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import data_path, load_config, pair_of, trade_cost
from rules_test import resample, test_windows

VARIANTS = [  # (lookback days, rebalance every N days, go to cash when the best coin is down)
    (14, 7, True), (30, 7, True), (60, 7, True), (90, 7, True),
    (30, 1, True), (30, 7, False),
]


def daily_closes(cfg):
    """Daily closes of every coin in train_symbols, one column per coin (NaN before listing)."""
    cols = {}
    for sym in cfg["train_symbols"]:
        try:
            d = resample(pd.read_parquet(data_path(cfg, sym)), "1D")
        except FileNotFoundError:
            print(f"warning: no data for {sym} - run `python -m data.fetch`")
            continue
        cols[sym] = d.set_index("open_time")["close"]
    return pd.DataFrame(cols).sort_index()


def run_rotation(hist, start_day, lookback, every, cash_filter, cost):
    """hist: daily closes up to the period's end (earlier days are only used for the lookback).
    Starts in cash on start_day. Returns (equity curve over the period, switches, % of days held)."""
    mom = hist / hist.shift(lookback) - 1               # return over the lookback, known at the close
    days = hist.index[hist.index >= start_day]
    holding, value, curve, switches, held = None, 1.0, [], 0, []
    for k, day in enumerate(days):
        if k % every == 0:                              # decision day: pick the leader
            m = mom.loc[day].dropna()
            target = None
            if len(m):
                best = m.idxmax()
                target = best if (m[best] > 0 or not cash_filter) else None
            if target != holding:
                value *= (1 - cost) if holding else 1.0  # sell the old coin
                value *= (1 - cost) if target else 1.0   # buy the new one
                switches += 1
                holding = target
        curve.append(value)
        held.append(holding or "cash")
        if k + 1 < len(days) and holding:                # earn the next day's move
            value *= hist.at[days[k + 1], holding] / hist.at[day, holding]
    return np.array(curve), switches, pd.Series(held).value_counts(normalize=True) * 100


def max_dd(curve):
    curve = np.asarray(curve, float)
    return (curve / np.maximum.accumulate(curve) - 1).min() * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    windows = test_windows(cfg, args.folds)
    folds = len(windows)
    cost, entry = trade_cost(cfg), cfg["fee"] + cfg["slippage"]
    closes = daily_closes(cfg)
    btc = cfg["symbol"]
    coins = list(closes.columns)
    print(f"coins: {', '.join(coins)}   cost per trade {cost * 100:.2f}% (a switch = sell + buy)")

    rows, mix = [], {}
    for i, t0, t1 in windows:
        hist = closes[closes.index <= t1]                       # nothing after the period
        per = hist[hist.index >= t0.normalize()]
        start_day = per.index[0]
        c_btc = per[btc]
        bh_btc = c_btc.values / c_btc.iloc[0] * (1 - entry)
        avail = [c for c in coins if not np.isnan(per[c].iloc[0])]
        eqw = np.mean([per[c].values / per[c].iloc[0] for c in avail], axis=0) * (1 - entry)
        base = {"period": i, "start": str(start_day.date()), "end": str(per.index[-1].date()),
                "btc_%": (bh_btc[-1] - 1) * 100, "btc_dd_%": max_dd(bh_btc),
                "equal_%": (eqw[-1] - 1) * 100, "equal_dd_%": max_dd(eqw), "equal_coins": len(avail)}
        for lb, every, cf in VARIANTS:
            name = f"{lb}-day leader, {'weekly' if every == 7 else 'daily'}{'' if cf else ', no cash'}"
            curve, sw, held = run_rotation(hist, start_day, lb, every, cf, cost)
            rows.append({**base, "variant": name, "rot_%": (curve[-1] - 1) * 100, "rot_dd_%": max_dd(curve),
                         "switches": sw})
            mix.setdefault(name, []).append(held)
    res = pd.DataFrame(rows)

    out = Path(cfg["paths"]["report"])
    out.mkdir(parents=True, exist_ok=True)
    res.round(2).to_csv(out / "rotation_by_period.csv", index=False)

    chain = lambda x: (np.prod(1 + np.asarray(x) / 100) - 1) * 100
    one = res[res.variant == res.variant.iloc[0]]
    btc_chain, eq_chain = chain(one["btc_%"]), chain(one["equal_%"])
    print(f"\n================ COIN ROTATION vs BUY & HOLD ({folds} walk-forward periods) ================")
    print(f"{'':32}{'chained':>9}{'beat BTC':>10}{'beat all3':>11}{'worst drop':>12}{'switches':>10}   time held")
    for name, g in res.groupby("variant", sort=False):
        held = pd.concat(mix[name], axis=1).fillna(0).mean(axis=1)
        held_s = "  ".join(f"{k.split('/')[0]} {v:.0f}%" for k, v in held.sort_index().items())
        print(f"{name:32}{chain(g['rot_%']):+8.0f}%{int((g['rot_%'] > g['btc_%']).sum()):>6}/{folds}"
              f"{int((g['rot_%'] > g['equal_%']).sum()):>8}/{folds}{g['rot_dd_%'].min():>11.0f}%"
              f"{g['switches'].mean():>10.0f}   {held_s}")
    print(f"{'buy & hold BTC':32}{btc_chain:+8.0f}%{'':21}{one['btc_dd_%'].min():>11.0f}%")
    print(f"{'buy & hold all coins equally':32}{eq_chain:+8.0f}%{'':21}{one['equal_dd_%'].min():>11.0f}%")

    print("\nper period (rotation = '30-day leader, weekly'):")
    show = res[res.variant == "30-day leader, weekly"][["period", "start", "end", "rot_%", "btc_%", "equal_%",
                                                         "rot_dd_%", "btc_dd_%", "switches"]]
    print(show.round(1).to_string(index=False))

    summ = res.groupby("variant", sort=False).apply(
        lambda g: pd.Series({"chained": chain(g["rot_%"]), "wins_btc": (g["rot_%"] > g["btc_%"]).sum()}),
        include_groups=False)
    good = summ[(summ["chained"] > btc_chain) & (summ["wins_btc"] >= folds * 0.67)]
    beat_some = summ[summ["chained"] > btc_chain]
    if len(good) >= len(summ) / 2:
        verdict = "✅ most variants beat buy & hold BTC, consistently -> rotation looks like a real signal"
    elif len(beat_some):
        verdict = (f"🤔 {len(beat_some)}/{len(summ)} variants beat buy & hold BTC overall, but not consistently "
                   "- could be luck; check the per-period table and worst drops")
    else:
        verdict = "❌ no rotation variant beats buy & hold BTC on these periods"
    print(f"\nverdict: {verdict}")
    print(f"details -> {out / 'rotation_by_period.csv'}")


if __name__ == "__main__":
    main()
