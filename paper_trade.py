"""Paper trading: run the trained agent with FAKE money and watch it in the dashboard's Live tab.

  python paper_trade.py             # live: real Kraken prices, one decision per closed bar
  python paper_trade.py --replay    # demo: replays the held-out test data bar by bar, fast
  python paper_trade.py --reset     # start over with fresh fake cash (combine with either mode)

No API key needed and no real orders are ever sent - this only reads public prices.
State is saved after every bar, so you can stop (Ctrl+C) and restart and it resumes.
"""
import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agents import load_agent
import numpy as np

from common import live_observation, load_config, load_norm, required_history



def fmt_ts(ts_ms):
    return pd.to_datetime(ts_ms, unit="ms").strftime("%Y-%m-%d %H:%M")


class PaperAccount:
    """Fake-money spot account: all-in BTC (long) or all-in USD (flat)."""

    EQ_FIELDS = ["time", "open", "high", "low", "close", "price", "equity", "buy_hold",
                 "position", "action"]
    TR_FIELDS = ["time", "side", "price", "qty_btc", "fee_usd", "equity_after"]

    def __init__(self, folder, starting_cash, fee, reset=False, symbol="BTC/USD"):
        self.dir = Path(folder)
        if reset and self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fee = fee
        self.state_f = self.dir / "state.json"
        self.eq_f, self.tr_f = self.dir / "equity.csv", self.dir / "trades.csv"
        if self.eq_f.exists() and self.eq_f.read_text().split("\n", 1)[0] != ",".join(self.EQ_FIELDS):
            raise SystemExit("Log files are from an older version - rerun with --reset to start fresh.")
        self.base, self.cur = symbol.split("/")
        if self.state_f.exists():
            self.s = json.loads(self.state_f.read_text())
            if self.s.get("symbol", "BTC/USD") != symbol:
                raise SystemExit(f"This paper account trades {self.s.get('symbol', 'BTC/USD')}, but the "
                                 f"config now says {symbol} - rerun with --reset to start fresh.")
            print(f"resuming: {self.s['cash']:.2f} {self.cur} cash + {self.s['btc']:.8f} {self.base}, "
                  f"{self.s['n_trades']} trades so far")
        else:
            self.s = {"symbol": symbol, "currency": self.cur,
                      "cash": float(starting_cash), "btc": 0.0, "pos": 0,
                      "start_cash": float(starting_cash), "start_price": None,
                      "last_bar_ts": None, "n_trades": 0,
                      "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            for f, fields in ((self.eq_f, self.EQ_FIELDS), (self.tr_f, self.TR_FIELDS)):
                with open(f, "w", newline="") as fh:
                    csv.writer(fh).writerow(fields)

    def _append(self, f, row):
        with open(f, "a", newline="") as fh:
            csv.writer(fh).writerow(row)

    def step(self, bar, action, bid, ask, votes=""):
        """bar = the candle just closed (ts, open, high, low, close); trade happens at bid/ask now."""
        bar_ts = int(bar["ts"])
        s, mid = self.s, (bid + ask) / 2
        pos_before = s["pos"]
        if s["start_price"] is None:
            s["start_price"] = ask
        t = fmt_ts(bar_ts)
        if action == 1 and s["pos"] == 0:            # BUY with all cash, at the ask
            fee_usd = s["cash"] * self.fee
            qty = (s["cash"] - fee_usd) / ask
            s["btc"], s["cash"], s["pos"] = qty, 0.0, 1
            s["n_trades"] += 1
            self._append(self.tr_f, [t, "BUY", round(ask, 2), round(qty, 8), round(fee_usd, 4),
                                     round(self.equity(mid), 4)])
        elif action == 0 and s["pos"] == 1:          # SELL all BTC, at the bid
            gross = s["btc"] * bid
            fee_usd = gross * self.fee
            qty = s["btc"]
            s["cash"], s["btc"], s["pos"] = gross - fee_usd, 0.0, 0
            s["n_trades"] += 1
            self._append(self.tr_f, [t, "SELL", round(bid, 2), round(qty, 8), round(fee_usd, 4),
                                     round(self.equity(mid), 4)])
        bh = s["start_cash"] * (1 - self.fee) * mid / s["start_price"]
        ohlc = [round(float(bar[k]), 2) for k in ("open", "high", "low", "close")]
        self._append(self.eq_f, [t, *ohlc, round(mid, 2), round(self.equity(mid), 4),
                                 round(bh, 4), s["pos"], action])
        # position context the agent sees next bar (same bookkeeping as the training env)
        if s["pos"]:
            s["held"] = s.get("held", 0) + 1 if pos_before else 1
            if not pos_before:
                s["entry_ref"] = float(bar["close"])
        else:
            s["held"], s["entry_ref"] = 0, None
        s["last_bar_ts"] = int(bar_ts)
        s["votes"] = votes  # ensemble agreement, e.g. "2/3 long" (shown on the dashboard)
        self.state_f.write_text(json.dumps(s, indent=1))
        return self.equity(mid), bh

    def equity(self, price):
        return self.s["cash"] + self.s["btc"] * price


def decide(model, norm, cfg, bars, acct):
    s = acct.s
    upnl = np.log(float(bars["close"].iloc[-1]) / s["entry_ref"]) if s["pos"] and s.get("entry_ref") else 0.0
    obs = live_observation(bars, norm, cfg["symbol"], cfg["train"]["window"], s["pos"],
                           upnl, s.get("held", 0))
    if obs is None:
        return None
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def log_line(bar_ts, price, action, acct, eq, bh, votes=""):
    pos = f"LONG {acct.base:<4}" if acct.s["pos"] else "CASH     "
    v = f"  votes {votes}" if votes else ""
    print(f"{fmt_ts(bar_ts)}  price {price:>10.2f}  action {'long' if action else 'flat'}{v}  "
          f"-> {pos}  equity {eq:7.2f} {acct.cur}  (buy&hold {bh:7.2f})  trades {acct.s['n_trades']}")


def run_replay(cfg, model, norm, acct, delay, max_bars):
    """Feed the held-out test period one bar at a time, as if it were happening live."""
    df = pd.read_parquet(cfg["paths"]["data"])
    start = int(df.index[df["ts"] == norm["test_start_ts"]][0])
    if acct.s["last_bar_ts"] is not None:  # resume
        start = max(start, int(df.index[df["ts"] == acct.s["last_bar_ts"]][0]) + 1)
    end = len(df) if max_bars is None else min(len(df), start + max_bars)
    slip = cfg["slippage"]
    hist = required_history(cfg)
    print(f"replaying {end - start} bars of {cfg['timeframe']} "
          f"({fmt_ts(df['ts'].iloc[start])} -> {fmt_ts(df['ts'].iloc[end - 1])})")
    for i in range(start, end):
        bars = df.iloc[max(0, i - hist + 1): i + 1]
        action = decide(model, norm, cfg, bars, acct)
        if action is None:
            continue
        price = df["close"].iloc[i]
        eq, bh = acct.step(df.iloc[i], action, price * (1 - slip), price * (1 + slip),
                           model.votes_str())
        log_line(df["ts"].iloc[i], price, action, acct, eq, bh, model.votes_str())
        time.sleep(delay)


def run_live(cfg, model, norm, acct):
    import ccxt

    ex = getattr(ccxt, cfg["exchange"])({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(cfg["timeframe"]) * 1000
    hist = required_history(cfg)
    if cfg["exchange"] == "kraken" and hist > 720:
        print(f"WARNING: features need {hist} candles but Kraken only returns 720 - "
              "use a bigger timeframe (15m or 1h).")
    sym = cfg["live_symbol"]
    ex.load_markets()
    if sym not in ex.markets:
        raise SystemExit(f"{cfg['exchange']} doesn't list {sym} - check live_symbol in config.yaml")
    if sym != cfg["symbol"]:
        print(f"model trained on {cfg['symbol']} data, trading {sym} (same % moves, different currency)")
    print(f"paper trading {sym} on {cfg['exchange']} ({cfg['timeframe']} bars) - Ctrl+C to stop")
    while True:
        try:
            raw = ex.fetch_ohlcv(sym, cfg["timeframe"], limit=hist)
            bars = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            bars = bars[bars["ts"] + tf_ms <= ex.milliseconds()].tail(hist)  # closed bars only
            bar_ts = int(bars["ts"].iloc[-1])
            if bar_ts != acct.s["last_bar_ts"]:
                action = decide(model, norm, cfg, bars, acct)
                if action is None:
                    print("not enough history yet, waiting...")
                else:
                    tk = ex.fetch_ticker(sym)
                    last = tk.get("last") or bars["close"].iloc[-1]
                    bid, ask = tk.get("bid") or last, tk.get("ask") or last
                    eq, bh = acct.step(bars.iloc[-1], action, bid, ask, model.votes_str())
                    log_line(bar_ts, (bid + ask) / 2, action, acct, eq, bh, model.votes_str())
        except ccxt.BaseError as e:
            print(f"exchange error ({type(e).__name__}): {e} - retrying in 30s")
            time.sleep(30)
            continue
        # sleep until the next bar closes (+10s so the exchange has finalized it)
        now = ex.milliseconds()
        wait = ((now // tf_ms + 1) * tf_ms - now) / 1000 + 10
        print(f"  next decision in {wait / 60:.1f} min")
        time.sleep(wait)


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--replay", action="store_true", help="replay held-out test data instead of live")
    p.add_argument("--reset", action="store_true", help="start over with fresh fake cash")
    p.add_argument("--delay", type=float, default=cfg["paper"]["replay_delay"],
                   help="seconds per bar in replay mode")
    p.add_argument("--bars", type=int, default=None, help="max bars to replay")
    args = p.parse_args()

    model = load_agent(cfg)
    norm = load_norm(cfg)
    mode = "replay" if args.replay else "live"
    # replay uses the training data (symbol); live uses the live pair (e.g. BTC/CAD)
    acct = PaperAccount(Path(cfg["paths"]["report"]) / f"paper_{mode}",
                        cfg["paper"]["starting_cash"], cfg["fee"], reset=args.reset,
                        symbol=cfg["live_symbol"] if mode == "live" else cfg["symbol"])
    try:
        if args.replay:
            run_replay(cfg, model, norm, acct, args.delay, args.bars)
        else:
            run_live(cfg, model, norm, acct)
    except KeyboardInterrupt:
        print("\nstopped - state saved, run again to resume")


if __name__ == "__main__":
    main()
