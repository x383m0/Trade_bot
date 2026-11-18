"""Download OHLCV candles with ccxt and save to parquet.

Usage: python -m data.fetch [--synthetic]
Downloads from `data_exchange` in config.yaml (Coinbase by default: years of public history).
--synthetic generates fake random-walk data (for testing the pipeline offline).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config

# max candles per request for exchanges that cap it
LIMITS = {"coinbase": 300, "kraken": 720, "bitstamp": 1000, "ndax": 1000}


def fetch_ohlcv(cfg):
    import ccxt

    name = cfg["data_exchange"]
    if name == "kraken":
        print("WARNING: Kraken's API only returns the last ~720 candles - use coinbase for history.")
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    ex.load_markets()
    tf_ms = ex.parse_timeframe(cfg["timeframe"]) * 1000
    limit = LIMITS.get(name, 1000)
    now = ex.milliseconds()
    since = now - int(cfg["years"] * 365 * 24 * 3600 * 1000)
    rows = []
    while since < now:
        batch = ex.fetch_ohlcv(cfg["symbol"], cfg["timeframe"], since=since, limit=limit)
        if batch:
            rows += batch
            since = batch[-1][0] + tf_ms
        else:  # gap with no data - skip ahead
            since += limit * tf_ms
        print(f"\rfetched {len(rows)} candles "
              f"(up to {pd.to_datetime(since, unit='ms'):%Y-%m-%d})", end="")
    print()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df[df["ts"] + tf_ms <= now]  # drop the still-forming candle
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def synthetic_ohlcv(n=60000, seed=0, freq="15min"):
    """Random walk with volatility clustering. NO real edge - pipeline testing only."""
    rng = np.random.default_rng(seed)
    vol = 0.002 * np.exp(0.5 * np.cumsum(rng.normal(0, 0.05, n)) / np.sqrt(np.arange(1, n + 1)))
    ret = rng.normal(0, 1, n) * vol
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, vol)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(3, 0.5, n)
    dt = pd.date_range("2022-01-01", periods=n, freq=freq)
    ts = ((dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)).astype("int64")  # ms, unit-safe
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    if args.synthetic:
        tf = cfg["timeframe"]
        df = synthetic_ohlcv(freq=tf[:-1] + {"m": "min", "h": "h", "d": "D"}[tf[-1]])
    else:
        df = fetch_ohlcv(cfg)
    out = Path(cfg["paths"]["data"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"saved {len(df)} rows -> {out}{'  (SYNTHETIC)' if args.synthetic else ''}")


if __name__ == "__main__":
    main()
