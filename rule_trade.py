"""Paper trade the 200-day rule with FAKE money (no model, no training).

  python rule_trade.py              # live: real Kraken daily candles, one decision per day
  python rule_trade.py --replay     # demo: replays the final-exam period day by day
  python rule_trade.py --reset      # start over with fresh fake cash (combine with either mode)

The rule (exactly the one tested in rules_test.py / info_test.py):
  every day at the daily close (00:00 UTC), compare the price with its 200-day average:
    price above the average + 2%   -> hold BTC
    price below the average - 2%   -> cash
    in between                     -> keep doing what it was doing (the 2% buffer stops it from
                                      flipping back and forth when price hugs the average)
It isn't trying to beat buy & hold. It tries to sit out the big crashes: in the tests it made
a bit less than holding, with crashes about half as deep.

Orders: a limit order at the daily close (maker fee from config.yaml, orders.maker_fee). It fills if
the next day trades through that price; if it doesn't, the rule just places it again the day after.
min_hold / stop_loss from config.yaml are NOT used: the rule is tested without them.

Runs alongside paper_trade.py (separate account folder). Watch it in the dashboard's Live tab
("200-day rule"). Live mode can stay running for months. If it was stopped, it catches up on the
days it missed when restarted (if the order would have filled, it fills at that day's price).
No API key needed and no real orders are ever sent: this only reads public prices.
"""
import argparse
import csv
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config
from info_test import final_window
from paper_trade import PaperAccount, fmt_ts
from rules_test import signals

RULE = "200-day avg, 2% buffer"
DAY_MS = 86_400_000
MIN_DAYS = 200
RULE_FIELDS = ["time", "ma200", "buy_line", "sell_line"]


def daily_ohlc(df1h):
    """Daily candles (UTC days) built from 1h candles: ts = day start in ms, like exchange 1d candles."""
    t = pd.to_datetime(df1h["ts"], unit="ms")
    g = df1h.set_index(t).resample("1D")
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last()}).dropna()
    d["ts"] = (d.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    return d.reset_index(drop=True)[["ts", "open", "high", "low", "close"]]


def decide(closes):
    """(position the rule wants, 200-day average) from daily closes up to and including today.
    The position follows the WHOLE history (in the buffer zone it keeps its last state), so a fresh
    start agrees with the backtest."""
    closes = pd.Series(np.asarray(closes, dtype=float))
    if len(closes) < MIN_DAYS:
        return None, None
    want = int(signals(closes, 1)[RULE].iloc[-1])
    return want, float(closes.tail(MIN_DAYS).mean())


class RuleAccount(PaperAccount):
    """The paper account from paper_trade.py, plus a log of the 200-day line for the chart."""

    def __init__(self, folder, cash, cfg, reset, symbol):
        cfg = copy.deepcopy(cfg)
        cfg["rules"] = {"min_hold": 0, "stop_loss": 0.0}       # the rule was tested without them
        super().__init__(folder, cash, cfg, reset=reset, symbol=symbol)
        self.s.update({"strategy": "200-day rule", "timeframe": "1d"})
        self.rule_f = self.dir / "rule.csv"
        if not self.rule_f.exists():
            with open(self.rule_f, "w", newline="") as fh:
                csv.writer(fh).writerow(RULE_FIELDS)

    def day(self, bar, closes, slippage, quiet=False):
        """One daily close: fill yesterday's order if it traded through, decide, place the order."""
        want, ma = decide(closes)
        if want is None:
            print(f"{fmt_ts(int(bar['ts']))[:10]}  only {len(closes)} days of history - need {MIN_DAYS}")
            return
        self.begin_bar(bar)
        close = float(bar["close"])
        self._append(self.rule_f, [fmt_ts(int(bar["ts"])), round(ma, 2), round(ma * 1.02, 2),
                                   round(ma * 0.98, 2)])
        eq, bh = self.act(bar, want, close * (1 - slippage), close * (1 + slippage))
        if quiet:
            return
        dist = (close / ma - 1) * 100
        order = self.s.get("pending")
        todo = f"order: {order['side']} at {order['price']:,.2f}" if order else "no order"
        print(f"{fmt_ts(int(bar['ts']))[:10]}  close {close:>11,.2f}  200-day avg {ma:>11,.2f} ({dist:+5.1f}%)"
              f"  rule says {'BTC ' if want else 'CASH'}  -> holding {'BTC ' if self.s['pos'] else 'CASH'}"
              f"  {todo:<26}  equity {eq:7.2f} {self.cur} (buy&hold {bh:7.2f})  trades {self.s['n_trades']}")


def run_replay(cfg, acct, delay, max_days):
    daily = daily_ohlc(pd.read_parquet(cfg["paths"]["data"]))
    t0, _ = final_window(cfg)
    start = int(np.searchsorted(daily["ts"].values, int(t0.normalize().value // 1_000_000)))
    if acct.s["last_bar_ts"] is not None:      # resume
        start = max(start, int(np.searchsorted(daily["ts"].values, acct.s["last_bar_ts"])) + 1)
    end = len(daily) if max_days is None else min(len(daily), start + max_days)
    print(f"replaying {end - start} days ({fmt_ts(daily['ts'].iloc[start])[:10]} -> "
          f"{fmt_ts(daily['ts'].iloc[end - 1])[:10]}), the final-exam period")
    for i in range(start, end):
        acct.day(daily.iloc[i], daily["close"].values[:i + 1], cfg["slippage"])
        time.sleep(delay)


def run_live(cfg, acct):
    import ccxt

    ex = getattr(ccxt, cfg["exchange"])({"enableRateLimit": True})
    sym = cfg["live_symbol"]
    ex.load_markets()
    if sym not in ex.markets:
        raise SystemExit(f"{cfg['exchange']} doesn't list {sym} - check live_symbol in config.yaml")
    print(f"200-day rule, paper trading {sym} on {cfg['exchange']}: one decision per day at 00:00 UTC "
          "- Ctrl+C to stop")
    while True:
        try:
            raw = ex.fetch_ohlcv(sym, "1d", limit=720)
            d = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            d = d[d["ts"] + DAY_MS <= ex.milliseconds()].reset_index(drop=True)     # finished days only
            last = acct.s["last_bar_ts"]
            new = d.index[d["ts"] > last] if last is not None else d.index[-1:]
            if last is not None and len(new) > 1:
                print(f"catching up on {len(new)} days missed while stopped")
            for i in new:
                acct.day(d.iloc[i], d["close"].values[:i + 1], cfg["slippage"])
        except ccxt.BaseError as e:
            print(f"exchange error ({type(e).__name__}): {e} - retrying in 60s")
            time.sleep(60)
            continue
        # wait for the next daily close (+2 min so the exchange has finalized the candle). Sleeping in
        # short steps against the real clock still works if the computer went to sleep in between.
        wake = (ex.milliseconds() // DAY_MS + 1) * DAY_MS + 120_000
        print(f"  next decision at {fmt_ts(wake)} UTC "
              f"(in {(wake - ex.milliseconds()) / 3_600_000:.1f} h) - you can leave this running")
        while ex.milliseconds() < wake:
            time.sleep(min(60, max(1, (wake - ex.milliseconds()) / 1000)))


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", action="store_true", help="replay the final-exam period instead of live")
    p.add_argument("--reset", action="store_true", help="start over with fresh fake cash")
    p.add_argument("--delay", type=float, default=0.1, help="seconds per day in replay mode")
    p.add_argument("--days", type=int, default=None, help="max days to replay")
    args = p.parse_args()

    mode = "replay" if args.replay else "live"
    acct = RuleAccount(Path(cfg["paths"]["report"]) / f"paper_rule_{mode}", cfg["paper"]["starting_cash"],
                       cfg, reset=args.reset, symbol=cfg["live_symbol"] if mode == "live" else cfg["symbol"])
    try:
        if args.replay:
            run_replay(cfg, acct, args.delay, args.days)
        else:
            run_live(cfg, acct)
    except KeyboardInterrupt:
        print("\nstopped - state saved, run again to resume")


if __name__ == "__main__":
    main()
