"""Does information that prices DON'T contain help? Fear & Greed + funding rate rules vs buy & hold.

Usage:
  python -m data.fetch_extra      # first: download Fear & Greed + funding history
  python info_test.py

All rules decide once a day (daily close, 00:00 UTC) on BTC, long or cash, using only data from
the previous day or earlier (no peeking). Same cost per switch as the bot.

The rules were fixed BEFORE any result was seen, from common trading wisdom:
  1. greed exit           hold BTC; cash at extreme greed (F&G >= 80); buy back when F&G <= 50
  2. fear buyer           cash; buy at extreme fear (F&G <= 20); sell at extreme greed (F&G >= 80)
  3. crowded longs exit   hold BTC; cash when funding is in its top 10% of the past year;
                          buy back when funding is back at or below its 1-year median
  4. 200-day + greed exit     the 200-day rule (2% buffer), AND the greed exit
  5. 200-day + crowded exit   the 200-day rule (2% buffer), AND the crowded longs exit

Graded on the 6 walk-forward periods, against buy & hold BTC and against the 200-day rule alone
(the "smaller crashes" benchmark). The last column is the final-exam period (Oct 2025 -> now):
these rules were never run on it, but we HAVE seen that it was a crash, so treat it as a bonus
check, not proof.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_splits, trade_cost
from rules_test import resample, signals, simulate, test_windows

WIN_MARGIN = 0.5   # a period only counts as a win if it beats the benchmark by more than 0.5 points
RULES = ["greed exit", "fear buyer", "crowded longs exit", "200-day + greed exit", "200-day + crowded exit"]


def hysteresis(enter, leave, start=0):
    """1 from a day where `enter` is true until a day where `leave` is true (then 0 again)."""
    pos, out = start, []
    for e, l in zip(enter.values, leave.values):
        if pos == 0 and e:
            pos = 1
        elif pos == 1 and l:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=enter.index)


def build_signals(btc, fgi, funding):
    """btc: daily closes indexed by day. fgi / funding: daily series indexed by day (raw).
    Returns {rule: 0/1 series}; every input is shifted one day (only yesterday is known)."""
    days = btc.index
    f = fgi.reindex(days).ffill(limit=3).shift(1)                 # yesterday's Fear & Greed
    fu = funding.reindex(days).shift(1)                            # yesterday's average funding
    hi = fu.rolling(365, min_periods=120).quantile(0.90)           # top-10% line of the past year
    med = fu.rolling(365, min_periods=120).median()
    ma = signals(btc, 1)["200-day avg, 2% buffer"]

    greed_ok = 1 - hysteresis(f >= 80, f <= 50)                    # 0 while "too greedy"
    crowd_ok = 1 - hysteresis(fu > hi, fu <= med)                  # 0 while longs are crowded
    no_funding = fu.isna() | hi.isna()
    crowd_ok = crowd_ok.where(~no_funding, 1)                      # no funding history yet -> no opinion
    sig = {
        "greed exit": greed_ok.where(f.notna(), 1),
        "fear buyer": hysteresis(f <= 20, f >= 80).where(f.notna(), 0),
        "crowded longs exit": crowd_ok,
        "200-day + greed exit": ma * greed_ok.where(f.notna(), 1),
        "200-day + crowded exit": ma * crowd_ok,
        "200-day rule": ma,
    }
    return {k: v.astype(int) for k, v in sig.items()}, no_funding


def final_window(cfg):
    splits, _ = load_splits(cfg)
    te = splits[cfg["symbol"]]["test"][0]
    t0, t1 = pd.to_datetime(te["ts"].iloc[[cfg["train"]["window"] - 1, -1]], unit="ms")
    return t0, t1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    try:
        fgi = pd.read_parquet("data/fear_greed.parquet").set_index("date")["fear_greed"]
        funding = pd.read_parquet("data/funding_btc.parquet").set_index("date")["funding"]
    except FileNotFoundError:
        raise SystemExit("run `python -m data.fetch_extra` first (downloads Fear & Greed + funding)")

    windows = test_windows(cfg, args.folds)
    folds = len(windows)
    t0f, t1f = final_window(cfg)
    windows_all = windows + [("final", t0f, t1f)]
    print(f"final-exam period: {t0f.date()} -> {t1f.date()}")
    print(f"Fear & Greed: {fgi.index.min().date()} -> {fgi.index.max().date()}   "
          f"funding: {funding.index.min().date()} -> {funding.index.max().date()}")

    daily = resample(pd.read_parquet(cfg["paths"]["data"]), "1D").set_index("open_time")["close"]
    sig, no_funding = build_signals(daily, fgi, funding)
    cost = trade_cost(cfg)
    entry = cost          # buy & hold pays the same cost as the rules (otherwise "just hold" looks like a win)

    rows = []
    for i, t0, t1 in windows_all:
        m = (daily.index >= t0.normalize()) & (daily.index <= t1)
        c = daily[m]
        bh = (c.iloc[-1] / c.iloc[0] * (1 - entry) - 1) * 100
        bh_dd = ((c / c.cummax()).min() - 1) * 100
        nf = float(no_funding[m].mean() * 100)
        for name, s in sig.items():
            ret, trades, inmkt, dd = simulate(c, s[m], cost)
            rows.append({"rule": name, "period": i, "return_%": ret, "buy_hold_%": bh, "bh_dd_%": bh_dd,
                         "trades": trades, "in_market_%": inmkt, "max_dd_%": dd, "no_funding_%": nf})
    res = pd.DataFrame(rows)
    out = Path(cfg["paths"]["report"])
    out.mkdir(parents=True, exist_ok=True)
    res.round(2).to_csv(out / "info_by_period.csv", index=False)

    chain = lambda x: (np.prod(1 + np.asarray(x) / 100) - 1) * 100
    wf = res[res.period != "final"]
    ma = wf[wf.rule == "200-day rule"].set_index("period")
    bh_ref = ma["buy_hold_%"]
    fin = res[res.period == "final"].set_index("rule")

    print(f"\n================ NEW INFORMATION RULES ({cfg['symbol']}, daily, {folds} walk-forward periods) ================")
    print(f"cost per switch {cost * 100:.2f}%   each rule decides with YESTERDAY's data only\n")
    hdr = (f"{'':26}{'chained':>9}{'beat B&H':>10}{'beat 200d':>11}{'worst drop':>12}{'in mkt':>8}"
           f"{'trades':>8}   |{'FINAL':>8}{'drop':>8}")
    print(hdr)
    summary = []
    for name in RULES + ["200-day rule"]:
        g = wf[wf.rule == name].set_index("period")
        ch = chain(g["return_%"])
        w_bh = int((g["return_%"] > bh_ref + WIN_MARGIN).sum())
        w_ma = int((g["return_%"] > ma["return_%"] + WIN_MARGIN).sum()) if name != "200-day rule" else None
        f = fin.loc[name]
        print(f"{name:26}{ch:+8.0f}%{w_bh:>6}/{folds}"
              + (f"{w_ma:>7}/{folds}" if w_ma is not None else f"{'':>9}")
              + f"{g['max_dd_%'].min():>11.0f}%{g['in_market_%'].mean():>7.0f}%{g['trades'].mean():>8.1f}"
              f"   |{f['return_%']:+7.0f}%{f['max_dd_%']:>7.0f}%")
        summary.append({"rule": name, "chained": ch, "wins_bh": w_bh, "wins_ma": w_ma,
                        "worst_dd": g["max_dd_%"].min()})
    print(f"{'buy & hold BTC':26}{chain(bh_ref):+8.0f}%{'':>20}{ma['bh_dd_%'].min():>11.0f}%{'':>16}"
          f"   |{fin.loc['200-day rule', 'buy_hold_%']:+7.0f}%{fin.loc['200-day rule', 'bh_dd_%']:>7.0f}%")

    nf = wf.groupby("period")["no_funding_%"].first()
    if nf.max() > 0:
        gaps = ", ".join(f"period {p}: {v:.0f}%" for p, v in nf.items() if v > 0)
        print(f"\nnote: funding rules had no funding history yet for part of: {gaps} "
              "(they just follow the other part of the rule / hold there)")

    s = pd.DataFrame(summary).set_index("rule")
    bh_chain, ma_row = chain(bh_ref), s.loc["200-day rule"]
    beats_bh = s.loc[RULES][(s.loc[RULES, "chained"] > bh_chain + WIN_MARGIN)
                            & (s.loc[RULES, "wins_bh"] >= folds * 0.67)]
    safer = s.loc[RULES][(s.loc[RULES, "chained"] > ma_row["chained"])
                         & (s.loc[RULES, "worst_dd"] >= ma_row["worst_dd"] - 2)
                         & (s.loc[RULES, "wins_ma"] >= folds / 2)]
    print()
    if len(beats_bh):
        print(f"verdict: ✅ {', '.join(beats_bh.index)} beat buy & hold consistently -> the new information "
              "has real signal: worth giving to the bot")
    elif len(safer):
        print(f"verdict: 🤔 {', '.join(safer.index)} improve the 200-day safety rule (more gain, crashes no "
              "worse) -> useful for the 'smaller crashes' goal, but don't beat buy & hold")
    else:
        print("verdict: ❌ no rule beats buy & hold or improves the 200-day rule -> this information doesn't "
              "help on these periods")
    print(f"details -> {out / 'info_by_period.csv'}")


if __name__ == "__main__":
    main()
