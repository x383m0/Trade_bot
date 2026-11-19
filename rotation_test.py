"""Coin rotation: hold whichever coin is strongest right now (or cash), on the walk-forward periods.

Usage:
  python rotation_test.py                      # BTC / ETH / SOL (train_symbols in config.yaml)
  python rotation_test.py --coins wide         # 11 coins - less hindsight in which coins we picked
  python rotation_test.py --coins BTC/USD ETH/USD XRP/USD
  python rotation_test.py --coins wide --final # the ONE-TIME final exam (see below)

--coins wide = BTC ETH SOL XRP ADA DOGE LTC LINK AVAX DOT BCH. Download them first:
  python -m data.fetch --symbol XRP/USD ADA/USD DOGE/USD LTC/USD LINK/USD AVAX/USD DOT/USD BCH/USD
Picking only today's winners (like SOL) flatters any rotation. A wider list, including coins
that did poorly, is a fairer test. (Still not perfect: coins that collapsed and were delisted,
like LUNA, aren't on Coinbase, so they can't be included.)

--final: runs the variant decided IN ADVANCE ("30-day leader, weekly", with cash) on the last
15% of history - the period nothing has been tested or tuned on. Run it once: if you change
things after seeing it and run again, it stops being a fair exam.

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

WIDE = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "LTC/USD", "LINK/USD",
        "AVAX/USD", "DOT/USD", "BCH/USD"]
FINAL_VARIANT = (30, 7, True)   # decided BEFORE the final exam was ever run - don't change it after

VARIANTS = [  # (lookback days, rebalance every N days, go to cash when the best coin is down)
    (14, 7, True), (30, 7, True), (60, 7, True), (90, 7, True),
    (30, 1, True), (30, 7, False),
]


def daily_closes(cfg, coins):
    """Daily closes, one column per coin (NaN before listing and on days with no trading)."""
    cols, missing = {}, []
    for sym in coins:
        try:
            d = resample(pd.read_parquet(data_path(cfg, sym)), "1D")
        except FileNotFoundError:
            missing.append(sym)
            continue
        cols[sym] = d.set_index("open_time")["close"]
    if missing:
        print(f"warning: no data for {', '.join(missing)} - skipping. Download with:\n"
              f"  python -m data.fetch --symbol {' '.join(missing)}")
    return pd.DataFrame(cols).sort_index()


def run_rotation(hist, start_day, lookback, every, cash_filter, cost):
    """hist: daily closes up to the period's end (earlier days are only used for the lookback).
    Starts in cash on start_day. Returns (equity curve over the period, switches, % of days held)."""
    mom = hist / hist.shift(lookback) - 1               # return over the lookback, known at the close
    px = hist.ffill()        # a coin with a missing day keeps its last price (can't be picked that day)
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
            value *= px.at[days[k + 1], holding] / px.at[day, holding]
    return np.array(curve), switches, pd.Series(held).value_counts(normalize=True) * 100


def max_dd(curve):
    curve = np.asarray(curve, float)
    return (curve / np.maximum.accumulate(curve) - 1).min() * 100


def final_exam(cfg, closes, cost, entry):
    """The pre-decided variant on the untouched last part of history (the normal --split test)."""
    from common import load_splits
    splits, norm = load_splits(cfg)
    te = splits[cfg["symbol"]]["test"][0]
    t0, t1 = pd.to_datetime(te["ts"].iloc[[cfg["train"]["window"] - 1, -1]], unit="ms")
    lb, every, cf = FINAL_VARIANT
    name = f"{lb}-day leader, {'weekly' if every == 7 else 'daily'}{'' if cf else ', no cash'}"
    hist = closes[closes.index <= t1]
    start_day = hist.index[hist.index >= t0.normalize()][0]
    per = hist[hist.index >= start_day].ffill()
    btc = cfg["symbol"]
    curve, sw, held = run_rotation(hist, start_day, lb, every, cf, cost)
    bh = per[btc].values / per[btc].iloc[0] * (1 - entry)
    avail = [c for c in closes.columns if not np.isnan(per[c].iloc[0])]
    eqw = np.mean([per[c].values / per[c].iloc[0] for c in avail], axis=0) * (1 - entry)
    rot, b, e = (curve[-1] - 1) * 100, (bh[-1] - 1) * 100, (eqw[-1] - 1) * 100
    print(f"\n================ FINAL EXAM: {start_day.date()} -> {per.index[-1].date()} ================")
    print(f"variant decided in advance: '{name}', coins: {', '.join(c.split('/')[0] for c in closes.columns)}")
    print("this period was never used to test or tune anything. Run it once.\n")
    print(f"{'':30}{'return':>9}{'worst drop':>12}")
    print(f"{'coin rotation':30}{rot:+8.1f}%{max_dd(curve):>11.1f}%   ({sw} switches)")
    print(f"{'buy & hold BTC':30}{b:+8.1f}%{max_dd(bh):>11.1f}%")
    print(f"{'buy & hold all coins equally':30}{e:+8.1f}%{max_dd(eqw):>11.1f}%   ({len(avail)} coins)")
    print("time held: " + "  ".join(f"{k.split('/')[0]} {v:.0f}%" for k, v in held.sort_index().items()))
    beat_b, beat_e = rot > b, rot > e
    print("\nverdict: " + ("✅ beat both buy & hold BTC and holding all coins" if beat_b and beat_e else
                         "🤔 beat one of the two" if beat_b or beat_e else "❌ lost to both"))
    print("one period is still one period - together with the walk-forward periods it tells the story")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--coins", nargs="+", default=None, metavar="COIN",
                    help="coins to rotate between, or 'wide' (default: train_symbols)")
    ap.add_argument("--final", action="store_true",
                    help="ONE-TIME final exam on the last 15%% of history, with the pre-decided variant")
    args = ap.parse_args()

    cfg = load_config()
    wanted = WIDE if args.coins == ["wide"] else (args.coins or cfg["train_symbols"])
    btc = cfg["symbol"]
    if btc not in wanted:
        wanted = [btc] + list(wanted)
    closes = daily_closes(cfg, wanted)
    coins = list(closes.columns)
    cost, entry = trade_cost(cfg), cfg["fee"] + cfg["slippage"]
    if args.final:
        return final_exam(cfg, closes, cost, entry)
    windows = test_windows(cfg, args.folds)
    folds = len(windows)
    print(f"coins: {', '.join(coins)}   cost per trade {cost * 100:.2f}% (a switch = sell + buy)")

    rows, mix = [], {}
    for i, t0, t1 in windows:
        hist = closes[closes.index <= t1]                       # nothing after the period
        per = hist[hist.index >= t0.normalize()].ffill()
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
    tag = "_wide" if args.coins == ["wide"] else ""
    res.round(2).to_csv(out / f"rotation_by_period{tag}.csv", index=False)

    chain = lambda x: (np.prod(1 + np.asarray(x) / 100) - 1) * 100
    one = res[res.variant == res.variant.iloc[0]]
    btc_chain, eq_chain = chain(one["btc_%"]), chain(one["equal_%"])
    print(f"\n================ COIN ROTATION vs BUY & HOLD ({folds} walk-forward periods) ================")
    print(f"{'':32}{'chained':>9}{'beat BTC':>10}{'beat equal':>11}{'worst drop':>12}{'switches':>10}   time held")
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
    print(f"details -> {out / f'rotation_by_period{tag}.csv'}")


if __name__ == "__main__":
    main()
