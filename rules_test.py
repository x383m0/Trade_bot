"""Simple trading rules vs buy & hold, on the SAME periods as the walk-forward test.

Usage:
  python rules_test.py                 # rules x timeframes (1h, 2h, 4h, 1d) on the 6 walk-forward periods

Why: if even a simple trend rule can't beat buy & hold on these periods, a neural network on the
same price data almost certainly won't either. If a rule DOES win, the signal exists - and the
column where it wins best tells you which candle size is worth training the bot on.

Each rule is long (hold the coin) or flat (cash). It decides at every candle close, using only
data up to that close, and pays the same cost per switch as the bot (config: orders / fee).
No training, so it takes a minute or two. Rules never touch the last 15% of history either.
Limit orders are assumed to always fill (a small advantage the bot doesn't get).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, load_splits, trade_cost
from walkforward import fold_windows

TIMEFRAMES = {"1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D"}


def test_windows(cfg, folds=None):
    """[(period, start, end)]: the exact test windows walkforward.py uses (same split code, data)."""
    wf = cfg.get("walkforward") or {}
    folds = folds or int(wf.get("folds", 6))
    start, end = float(wf.get("start", 0.40)), float(wf.get("end", 0.85))
    windows = []
    for i, fracs in enumerate(fold_windows(folds, start, end), 1):
        splits, _ = load_splits(cfg, fracs)
        te = splits[cfg["symbol"]]["test"][0]
        first = cfg["train"]["window"] - 1               # the bot starts trading after its window
        t0, t1 = pd.to_datetime(te["ts"].iloc[[first, -1]], unit="ms")
        windows.append((i, t0, t1))
        print(f"period {i}: {t0.date()} -> {t1.date()}")
    return windows


def resample(df1h, rule):
    """Candles of another size built from 1h candles (Kraken has no 2h, Coinbase no 4h)."""
    t = pd.to_datetime(df1h["ts"], unit="ms")
    kw = {} if rule.endswith("D") else {"origin": "epoch"}   # days already start at 00:00 UTC
    g = df1h.set_index(t).resample(rule, label="left", closed="left", **kw)
    out = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    out = out.dropna()
    out["open_time"] = out.index
    return out.reset_index(drop=True)


def signals(close, bars_per_day):
    """Long (1) / flat (0) for each candle, using only data up to that candle's close."""
    n = lambda days: max(1, int(round(days * bars_per_day)))
    ma50, ma200 = close.rolling(n(50)).mean(), close.rolling(n(200)).mean()
    band = []                                   # above/below the 200-day avg with a 2% buffer
    pos = 0
    for c, m in zip(close.values, ma200.values):
        if not np.isnan(m):
            if c > m * 1.02:
                pos = 1
            elif c < m * 0.98:
                pos = 0
        band.append(pos)
    return {
        "price above 200-day avg": (close > ma200).astype(int),
        "200-day avg, 2% buffer": pd.Series(band, index=close.index),
        "price above 50-day avg": (close > ma50).astype(int),
        "50-day avg above 200-day": (ma50 > ma200).astype(int),
        "30-day trend up": (close > close.shift(n(30))).astype(int),
    }


def simulate(close, pos, cost):
    """Hold `pos[k]` from close k to close k+1, paying `cost` on every switch (start in cash).
    Returns (return %, trades, % of time in market, max drawdown %)."""
    pos = pos.values.astype(float)
    r = np.log(close.values[1:] / close.values[:-1])
    held = pos[:-1]
    switches = np.abs(np.diff(np.concatenate([[0.0], held])))
    log_eq = np.cumsum(held * r - switches * cost)
    eq = np.exp(np.concatenate([[0.0], log_eq]))
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return (eq[-1] - 1) * 100, int(switches.sum()), held.mean() * 100, dd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    windows = test_windows(cfg, args.folds)
    folds = len(windows)
    cost, sym = trade_cost(cfg), cfg["symbol"]
    entry_cost = cost                                  # buy & hold pays the same cost as the rules

    df1h = pd.read_parquet(cfg["paths"]["data"])
    if cfg["timeframe"] != "1h":
        raise SystemExit("rules_test.py builds every candle size from 1h data: set timeframe: 1h")

    rows = []
    for tf, rule in TIMEFRAMES.items():
        bars = resample(df1h, rule)
        bpd = {"1h": 24, "2h": 12, "4h": 6, "1d": 1}[tf]
        sig = signals(bars["close"], bpd)
        for i, t0, t1 in windows:
            m = (bars["open_time"] >= t0) & (bars["open_time"] <= t1)
            c = bars.loc[m, "close"]
            bh = (c.iloc[-1] / c.iloc[0] * (1 - entry_cost) - 1) * 100
            bh_dd = ((c / c.cummax()).min() - 1) * 100
            for name, s in sig.items():
                ret, trades, inmkt, dd = simulate(c, s[m], cost)
                rows.append({"rule": name, "tf": tf, "period": i, "return_%": ret, "buy_hold_%": bh,
                             "trades": trades, "in_market_%": inmkt, "max_dd_%": dd, "bh_max_dd_%": bh_dd})
    res = pd.DataFrame(rows)

    # walk-forward bot, if it was run with these settings (for comparison)
    bot = None
    wf_sum = Path(cfg["paths"]["report"]) / "walkforward" / "summary.csv"
    if wf_sum.exists():
        b = pd.read_csv(wf_sum)
        if len(b) == folds:
            bot = b

    out = Path(cfg["paths"]["report"])
    out.mkdir(parents=True, exist_ok=True)
    res.round(2).to_csv(out / "rules_by_period.csv", index=False)

    def chain(x):
        return (np.prod(1 + np.asarray(x) / 100) - 1) * 100

    ref = res[(res.rule == res.rule.iloc[0]) & (res.tf == "1h")]
    bh_chain, bh_worst_dd = chain(ref["buy_hold_%"]), ref["bh_max_dd_%"].min()
    summ = []
    for (name, tf), g in res.groupby(["rule", "tf"], sort=False):
        summ.append({"rule": name, "tf": tf, "chained_%": chain(g["return_%"]),
                     "wins": int((g["return_%"] > g["buy_hold_%"] + 0.5).sum()),   # 0.5-point margin
                     "trades_per_period": g["trades"].mean(), "in_market_%": g["in_market_%"].mean(),
                     "worst_dd_%": g["max_dd_%"].min()})
    summ = pd.DataFrame(summ)
    summ.round(2).to_csv(out / "rules_summary.csv", index=False)

    # --- print: one row per rule, one column per candle size ---
    cell = lambda r: f"{r['chained_%']:+7.0f}% {r['wins']}/{folds} {r['trades_per_period']:5.0f}tr"
    print(f"\n================ SIMPLE RULES vs BUY & HOLD ({sym}, {folds} walk-forward periods) ================")
    print("each cell: all periods chained | periods that beat buy & hold | trades per period")
    print(f"cost per switch {cost * 100:.2f}% (same as the bot)\n")
    print(f"{'':28}" + "".join(f"{tf:>24}" for tf in TIMEFRAMES))
    for name in summ["rule"].unique():
        s = summ[summ.rule == name].set_index("tf")
        print(f"{name:28}" + "".join(f"{cell(s.loc[tf]):>24}" for tf in TIMEFRAMES))
    print(f"{'buy & hold':28}{bh_chain:+7.0f}%   (worst drop in a period {bh_worst_dd:.0f}%)")
    if bot is not None:
        print(f"{'your RL bot (walk-forward)':28}{chain(bot['bot_%']):+7.0f}% "
              f"{int((bot['bot_%'] > bot['buy_hold_%']).sum())}/{folds} {bot['trades'].mean():5.0f}tr   (1h)")

    best = summ.sort_values("chained_%", ascending=False).iloc[0]
    print(f"best: '{best['rule']}' on {best['tf']} candles: {best['chained_%']:+.0f}% chained, "
          f"beat buy & hold in {best['wins']}/{folds} periods, worst drop {best['worst_dd_%']:.0f}% "
          f"(buy & hold chained {bh_chain:+.0f}%)")
    if best["wins"] >= folds * 0.67 and best["chained_%"] > bh_chain:
        print("verdict: ✅ a simple rule beats buy & hold here -> the signal exists; that candle size "
              "is worth training the bot on")
    elif summ["chained_%"].max() > bh_chain or summ["wins"].max() >= folds / 2:
        print("verdict: 🤔 some rules win sometimes - look at worst drop too: a rule that matches buy & hold "
              "with much smaller crashes can still be worth it")
    else:
        print("verdict: ❌ no simple rule beats buy & hold on these periods - a bot on the same data "
              "is unlikely to either")
    print(f"\ndetails -> {out / 'rules_by_period.csv'}  and  {out / 'rules_summary.csv'}")


if __name__ == "__main__":
    main()
