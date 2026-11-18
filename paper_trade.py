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

import numpy as np

from agents import load_agent
from common import live_observation, load_config, load_norm, required_history, trade_rules
from env.trading_env import apply_rules, limit_filled


def fmt_ts(ts_ms):
    return pd.to_datetime(ts_ms, unit="ms").strftime("%Y-%m-%d %H:%M")


class PaperAccount:
    """Fake-money spot account: all-in on the coin (long) or all-in cash (flat).

    Follows the same rules as the training env (env/trading_env.py): market or limit orders,
    minimum holding time and stop-loss. Each new closed candle:
      1. begin_bar(): last candle's limit order fills if this candle traded through its price
         (otherwise it's cancelled); holding time ticks up.
      2. the agent decides, using position / open-trade P&L / holding time.
      3. act(): rules are applied, then the order is placed (market = now, limit = next candle).
    """

    VERSION = 2
    EQ_FIELDS = ["time", "open", "high", "low", "close", "price", "equity", "buy_hold",
                 "position", "action"]
    TR_FIELDS = ["time", "side", "order", "price", "qty_btc", "fee_usd", "equity_after"]

    def __init__(self, folder, starting_cash, cfg, reset=False, symbol="BTC/USD"):
        self.dir = Path(folder)
        if reset and self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fee, self.rules = cfg["fee"], trade_rules(cfg)
        self.state_f = self.dir / "state.json"
        self.eq_f, self.tr_f = self.dir / "equity.csv", self.dir / "trades.csv"
        for f, fields in ((self.eq_f, self.EQ_FIELDS), (self.tr_f, self.TR_FIELDS)):
            if f.exists() and f.read_text().split("\n", 1)[0] != ",".join(fields):
                raise SystemExit("Log files are from an older version - rerun with --reset to start fresh.")
        self.base, self.cur = symbol.split("/")
        if self.state_f.exists():
            self.s = json.loads(self.state_f.read_text())
            if self.s.get("v") != self.VERSION:
                raise SystemExit("Paper account is from an older version - rerun with --reset.")
            if self.s.get("symbol", "BTC/USD") != symbol:
                raise SystemExit(f"This paper account trades {self.s.get('symbol', 'BTC/USD')}, but the "
                                 f"config now says {symbol} - rerun with --reset to start fresh.")
            print(f"resuming: {self.s['cash']:.2f} {self.cur} cash + {self.s['btc']:.8f} {self.base}, "
                  f"{self.s['n_trades']} trades so far")
        else:
            self.s = {"v": self.VERSION, "symbol": symbol, "currency": self.cur,
                      "cash": float(starting_cash), "btc": 0.0, "pos": 0, "held": 0,
                      "entry_ref": None, "pending": None,
                      "start_cash": float(starting_cash), "start_price": None,
                      "last_bar_ts": None, "n_trades": 0,
                      "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            for f, fields in ((self.eq_f, self.EQ_FIELDS), (self.tr_f, self.TR_FIELDS)):
                with open(f, "w", newline="") as fh:
                    csv.writer(fh).writerow(fields)

    def _append(self, f, row):
        with open(f, "a", newline="") as fh:
            csv.writer(fh).writerow(row)

    def _trade(self, side, price, fee_rate, bar_ts, order, entry_ref=None):
        s = self.s
        if side == "BUY":
            fee = s["cash"] * fee_rate
            qty = (s["cash"] - fee) / price
            s["btc"], s["cash"], s["pos"], s["held"], s["entry_ref"] = qty, 0.0, 1, 0, entry_ref
        else:
            qty = s["btc"]
            gross = qty * price
            fee = gross * fee_rate
            s["cash"], s["btc"], s["pos"], s["held"], s["entry_ref"] = gross - fee, 0.0, 0, 0, None
        s["n_trades"] += 1
        self._append(self.tr_f, [fmt_ts(bar_ts), side, order, round(price, 2), round(qty, 8),
                                 round(fee, 4), round(self.equity(price), 4)])

    def begin_bar(self, bar):
        if self.s.get("begun_ts") == int(bar["ts"]):
            return   # already done for this candle (e.g. retry after a network error)
        self.s["begun_ts"] = int(bar["ts"])
        s, p = self.s, self.s.get("pending")
        if p:
            if limit_filled(p["side"] == "BUY", p["price"], float(bar["high"]), float(bar["low"]),
                            self.rules["fill_margin"]):
                self._trade(p["side"], p["price"], self.rules["maker_fee"], int(bar["ts"]), "limit",
                            entry_ref=p["price"])
            s["pending"] = None   # unfilled orders are cancelled; the agent decides again
        if s["pos"]:
            s["held"] += 1

    def upnl(self, close):
        s = self.s
        return float(np.log(close / s["entry_ref"])) if s["pos"] and s.get("entry_ref") else 0.0

    def act(self, bar, action, bid, ask, votes=""):
        """Apply the rules to the agent's decision and place the order. Returns (equity, buy&hold)."""
        s, mid, close, bar_ts = self.s, (bid + ask) / 2, float(bar["close"]), int(bar["ts"])
        if s["start_price"] is None:
            s["start_price"] = ask
        r = self.rules
        final, forced = apply_rules(action, s["pos"], s["held"], self.upnl(close),
                                    r["min_hold"], r["stop_loss"])
        if final != s["pos"]:
            side = "BUY" if final == 1 else "SELL"
            if forced or r["order_type"] == "market":
                self._trade(side, ask if side == "BUY" else bid, self.fee, bar_ts,
                            "stop-loss" if forced else "market", entry_ref=close)
            else:
                s["pending"] = {"side": side, "price": close}   # limit order at this candle's close
        bh = s["start_cash"] * (1 - self.fee) * mid / s["start_price"]
        ohlc = [round(float(bar[k]), 2) for k in ("open", "high", "low", "close")]
        self._append(self.eq_f, [fmt_ts(bar_ts), *ohlc, round(mid, 2), round(self.equity(mid), 4),
                                 round(bh, 4), s["pos"], action])
        s["last_bar_ts"] = bar_ts
        s["votes"] = votes  # ensemble agreement, e.g. "2/3 long" (shown on the dashboard)
        self.state_f.write_text(json.dumps(s, indent=1))
        return self.equity(mid), bh

    def equity(self, price):
        return self.s["cash"] + self.s["btc"] * price


def decide(model, norm, cfg, bars, acct):
    s = acct.s
    obs = live_observation(bars, norm, cfg["symbol"], cfg["train"]["window"], s["pos"],
                           acct.upnl(float(bars["close"].iloc[-1])), s.get("held", 0))
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
        acct.begin_bar(df.iloc[i])
        action = decide(model, norm, cfg, bars, acct)
        if action is None:
            continue
        price = df["close"].iloc[i]
        eq, bh = acct.act(df.iloc[i], action, price * (1 - slip), price * (1 + slip),
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
                acct.begin_bar(bars.iloc[-1])
                action = decide(model, norm, cfg, bars, acct)
                if action is None:
                    print("not enough history yet, waiting...")
                else:
                    tk = ex.fetch_ticker(sym)
                    last = tk.get("last") or bars["close"].iloc[-1]
                    bid, ask = tk.get("bid") or last, tk.get("ask") or last
                    eq, bh = acct.act(bars.iloc[-1], action, bid, ask, model.votes_str())
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
                        cfg["paper"]["starting_cash"], cfg, reset=args.reset,
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
