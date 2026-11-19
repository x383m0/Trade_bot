"""Grid bot vs buy & hold: are the "steady profits" claims true? Same periods as the walk-forward.

Usage:
  python grid_test.py

What a grid bot does: it puts buy orders on a ladder of prices below today's price and sell
orders above it. Each "rung" buys when price drops to it and sells one rung higher, so every
up-and-down wiggle earns the gap between rungs (minus fees). Bot apps advertise this as steady
profit. The catch they don't show: when price falls through the whole ladder, the bot ends up
holding all coin bought on the way down. When price rises above the ladder, it has sold
everything and sits in cash while the coin keeps climbing.

How it's simulated (spot, long only, like the Kraken / Binance / Pionex spot grids):
  - The ladder is centered on the price at the start of each period, rungs spaced evenly in %.
  - Money is split equally between rungs. Rungs above the price start with coin (bought at the
    start, paying the fee); rungs below start with cash.
  - Limit orders on 1h candles: a buy fills if the next candle's low trades THROUGH the rung by
    orders.fill_margin, a sell if the high does. Maker fee (orders.maker_fee) on every fill.
  - Each rung fills at most once per candle (we can't know the order of moves inside a candle,
    so no free round trips within one hour).

Variants, all fixed BEFORE any result was seen (common settings from bot apps):
  1. classic     +-20% range, 10 lines; stays put when price leaves the range
  2. tight       +-10% range, 10 lines (more trades, leaves its range more often)
  3. wide        +-40% range, 20 lines (rarely leaves its range, fewer trades)
  4. re-center   +-20%, 10 lines; when price leaves the range, rebuild the ladder around today's
                 price (buying / selling coin to rebalance, paying the fee)
  5. trailing up +-20%, 10 lines; re-centers only when price breaks ABOVE the range (app feature
                 "trailing up"), stays put below it
  6. 200-day + grid  classic grid while the 200-day rule (2% buffer) says "in"; all cash when it
                 says "out"; a fresh ladder when it says "in" again

Graded on the 6 walk-forward periods vs buy & hold (same fee, 0.5-point margin to count as a win).
FINAL column = the last 15% of history: never used to pick anything here, but we have already
seen it was a crash, so it's a bonus check, not proof.

"App profit" = what bot apps show as "grid profit": the gain from completed buy-low-sell-high
round trips only. It ignores what the coin still held is worth now - compare it with the real
result to see how the advertising works.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, trade_rules
from info_test import final_window
from rules_test import resample, signals, test_windows

WIN_MARGIN = 0.5
SIDEWAYS = 25.0     # a period is "sideways" if buy & hold moved less than +-25% (decided in advance)
VARIANTS = {        # name: (range +-, number of lines, what happens when price leaves the range)
    "classic":        (0.20, 10, "static"),
    "tight":          (0.10, 10, "static"),
    "wide":           (0.40, 20, "static"),
    "re-center":      (0.20, 10, "recenter"),
    "trailing up":    (0.20, 10, "trail_up"),
    "200-day + grid": (0.20, 10, "trend"),
}


class Grid:
    """One ladder: slot k buys at lo[k] and sells at hi[k] (= lo[k+1]). Each slot holds either
    cash (waiting to buy) or coin (waiting to sell)."""

    def __init__(self, price, equity, width, lines, cost, coin_value=0.0):
        levels = np.geomspace(price * (1 - width), price * (1 + width), lines)
        self.lo, self.hi = levels[:-1], levels[1:]
        self.bottom, self.top = levels[0], levels[-1]
        has_coin = self.lo >= price                       # rungs above the price start with coin
        target_coin = equity * has_coin.mean()
        equity -= cost * abs(target_coin - coin_value)    # buying / selling to set up the ladder
        s = equity / len(self.lo)
        self.coin = has_coin.copy()
        self.qty = np.where(has_coin, s / price, 0.0)
        self.cash = np.where(has_coin, 0.0, s)
        self.basis = np.full(len(self.lo), s)             # cash put into each rung's current trade
        self.app_profit = 0.0
        self.round_trips = 0

    def step(self, high, low, cost, margin):
        buy = ~self.coin & (low <= self.lo * (1 - margin))
        sell = self.coin & (high >= self.hi * (1 + margin))
        if buy.any():
            self.basis[buy] = self.cash[buy]
            self.qty[buy] = self.cash[buy] * (1 - cost) / self.lo[buy]
            self.cash[buy] = 0.0
            self.coin[buy] = True
        if sell.any():
            self.cash[sell] = self.qty[sell] * self.hi[sell] * (1 - cost)
            self.app_profit += float((self.cash[sell] - self.basis[sell]).sum())
            self.round_trips += int(sell.sum())
            self.qty[sell] = 0.0
            self.coin[sell] = False

    def value(self, price):
        return float(self.cash.sum() + self.qty.sum() * price)

    def coin_value(self, price):
        return float(self.qty.sum() * price)


def run_grid(bars, trend_in, width, lines, mode, cost, margin):
    """bars: 1h candles of one period. trend_in: 200-day rule (0/1) known at each candle.
    Returns a dict of results; equity starts at 1."""
    hi, lo, cl = bars["high"].values, bars["low"].values, bars["close"].values
    equity, grid, app_profit, trips, resets, outside = 1.0, None, 0.0, 0, 0, 0
    if mode != "trend" or trend_in[0]:
        grid = Grid(cl[0], equity, width, lines, cost)
    eq, coin_share = [1.0], [grid.coin_value(cl[0]) if grid else 0.0]
    for t in range(1, len(cl)):
        if grid is not None:
            grid.step(hi[t], lo[t], cost, margin)
            equity = grid.value(cl[t])
            above, below = cl[t] > grid.top, cl[t] < grid.bottom
            outside += above or below
            rebuild = (mode == "recenter" and (above or below)) or (mode == "trail_up" and above)
            if mode == "trend" and not trend_in[t]:                  # 200-day rule says out: sell all
                equity -= cost * grid.coin_value(cl[t])
                app_profit, trips = app_profit + grid.app_profit, trips + grid.round_trips
                grid = None
            elif rebuild:
                app_profit, trips, resets = app_profit + grid.app_profit, trips + grid.round_trips, resets + 1
                grid = Grid(cl[t], equity, width, lines, cost, coin_value=grid.coin_value(cl[t]))
        elif trend_in[t]:                                            # back in: fresh ladder
            grid = Grid(cl[t], equity, width, lines, cost)
            resets += 1
        if grid is not None:
            equity = grid.value(cl[t])
        eq.append(equity)
        coin_share.append(grid.coin_value(cl[t]) / equity if grid is not None and equity > 0 else 0.0)
    if grid is not None:
        app_profit, trips = app_profit + grid.app_profit, trips + grid.round_trips
    eq = np.asarray(eq)
    return {"return_%": (eq[-1] - 1) * 100,
            "max_dd_%": ((eq / np.maximum.accumulate(eq)).min() - 1) * 100,
            "app_profit_%": app_profit * 100, "round_trips": trips, "resets": resets,
            "outside_%": outside / max(1, len(cl) - 1) * 100,
            "in_coin_%": float(np.mean(coin_share)) * 100}


def trend_per_hour(df1h):
    """200-day rule (2% buffer) on daily closes, as known at each hourly candle: the decision made
    at a day's close applies from the next day on (no peeking)."""
    daily = resample(df1h, "1D").set_index("open_time")["close"]
    sig = signals(daily, 1)["200-day avg, 2% buffer"].shift(1).fillna(0)
    day = pd.to_datetime(df1h["ts"], unit="ms").dt.normalize()
    return sig.reindex(day.values).fillna(0).astype(int).values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if cfg["timeframe"] != "1h":
        raise SystemExit("grid_test.py uses 1h candles: set timeframe: 1h")
    rules = trade_rules(cfg)
    cost, margin = rules["maker_fee"], rules["fill_margin"]    # grid bots only use limit orders

    windows = test_windows(cfg, args.folds)
    folds = len(windows)
    t0f, t1f = final_window(cfg)
    print(f"final-exam period: {t0f.date()} -> {t1f.date()}")

    df = pd.read_parquet(cfg["paths"]["data"]).reset_index(drop=True)
    trend = trend_per_hour(df)
    times = pd.to_datetime(df["ts"], unit="ms")

    rows = []
    for i, t0, t1 in windows + [("final", t0f, t1f)]:
        m = ((times >= t0) & (times <= t1)).values
        bars = df[m]
        c = bars["close"].values
        bh = (c[-1] / c[0] * (1 - cost) - 1) * 100
        bh_dd = ((c / np.maximum.accumulate(c)).min() - 1) * 100
        for name, (width, lines, mode) in VARIANTS.items():
            r = run_grid(bars, trend[m], width, lines, mode, cost, margin)
            rows.append({"variant": name, "period": i, "buy_hold_%": bh, "bh_dd_%": bh_dd, **r})
    res = pd.DataFrame(rows)
    out = Path(cfg["paths"]["report"])
    out.mkdir(parents=True, exist_ok=True)
    res.round(2).to_csv(out / "grid_by_period.csv", index=False)

    chain = lambda x: (np.prod(1 + np.asarray(x) / 100) - 1) * 100
    wf, fin = res[res.period != "final"], res[res.period == "final"].set_index("variant")
    bh_ref = wf[wf.variant == "classic"].set_index("period")["buy_hold_%"]
    bh_dd = wf[wf.variant == "classic"]["bh_dd_%"].min()
    step = {n: ((1 + w) / (1 - w)) ** (1 / (l - 1)) * 100 - 100 for n, (w, l, _) in VARIANTS.items()}

    print(f"\n================ GRID BOT vs BUY & HOLD ({cfg['symbol']}, 1h, {folds} walk-forward periods) ================")
    print(f"maker fee {cost * 100:.2f}% per fill (a round trip pays it twice)   fill margin {margin * 100:.2f}%\n")
    print(f"{'':16}{'rung gap':>9}{'chained':>9}{'beat B&H':>10}{'worst drop':>12}{'in coin':>9}"
          f"{'trips':>7}{'outside':>9}   |{'FINAL':>8}{'drop':>8}")
    summary = {}
    for name in VARIANTS:
        g = wf[wf.variant == name].set_index("period")
        ch, wins = chain(g["return_%"]), int((g["return_%"] > bh_ref + WIN_MARGIN).sum())
        f = fin.loc[name]
        summary[name] = (ch, wins)
        print(f"{name:16}{step[name]:>8.1f}%{ch:+8.0f}%{wins:>6}/{folds}{g['max_dd_%'].min():>11.0f}%"
              f"{g['in_coin_%'].mean():>8.0f}%{g['round_trips'].mean():>7.0f}{g['outside_%'].mean():>8.0f}%"
              f"   |{f['return_%']:+7.0f}%{f['max_dd_%']:>7.0f}%")
    print(f"{'buy & hold BTC':16}{'':>9}{chain(bh_ref):+8.0f}%{'':>10}{bh_dd:>11.0f}%"
          f"{'':>25}   |{fin.loc['classic', 'buy_hold_%']:+7.0f}%{fin.loc['classic', 'bh_dd_%']:>7.0f}%")
    print("in coin = average share of the money held as BTC   trips = completed buy-low-sell-high per period"
          "\noutside = % of the time price was outside the ladder (the grid does nothing there)")

    # --- the advertising check: classic grid, period by period ---
    g = res[res.variant == "classic"]
    print(f"\n---- classic grid, period by period: what a bot app would show vs what really happened ----")
    print(f"{'period':>8}{'B&H':>9}{'market':>10}{'app profit':>12}{'real result':>13}{'vs B&H':>9}")
    for _, r in g.iterrows():
        kind = "sideways" if abs(r["buy_hold_%"]) < SIDEWAYS else ("up" if r["buy_hold_%"] > 0 else "down")
        print(f"{str(r['period']):>8}{r['buy_hold_%']:+8.0f}%{kind:>10}{r['app_profit_%']:+11.1f}%"
              f"{r['return_%']:+12.1f}%{r['return_%'] - r['buy_hold_%']:+8.0f}")

    # --- verdicts on the two claims (on the walk-forward periods; final is the bonus) ---
    gw = wf[wf.variant == "classic"]
    side = gw[gw["buy_hold_%"].abs() < SIDEWAYS]
    always_app = bool((gw["app_profit_%"] > 0).all())
    real_losses = int((gw["return_%"] < 0).sum())
    print()
    print(f"claim 1 'steady profit': app profit positive in {int((gw['app_profit_%'] > 0).sum())}/{folds} "
          f"periods, real result negative in {real_losses}/{folds}"
          + (" -> ❌ the 'profit' shown hides the losses on coin held" if always_app and real_losses
             else " -> ✅ no hidden losses" if not real_losses else ""))
    if len(side):
        sw = int((side["return_%"] > side["buy_hold_%"] + WIN_MARGIN).sum())
        print(f"claim 2 'wins in sideways markets': {sw}/{len(side)} sideways periods beat buy & hold "
              + ("-> ✅ true here" if sw > len(side) / 2 else "-> ❌ not even there"))
    else:
        print("claim 2 'wins in sideways markets': no sideways period (B&H within +-25%) to judge it")
    best = max(summary, key=lambda n: summary[n][0])
    ch, wins = summary[best]
    if wins >= folds * 0.67 and ch > chain(bh_ref):
        print(f"overall: ✅ '{best}' beats buy & hold consistently - worth paper trading")
    elif ch > chain(bh_ref) or wins >= folds / 2:
        print(f"overall: 🤔 '{best}' wins sometimes ({wins}/{folds}) - check its worst drop before trusting it")
    else:
        print(f"overall: ❌ no grid variant beats buy & hold (best: '{best}', {ch:+.0f}% vs {chain(bh_ref):+.0f}%)")
    print(f"details -> {out / 'grid_by_period.csv'}")


if __name__ == "__main__":
    main()
