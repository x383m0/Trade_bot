"""Check how much candle history exchanges have for a trading pair.

Usage: python -m data.check_history                  # BTC/CAD on several exchanges
       python -m data.check_history ETH/CAD
       python -m data.check_history BTC/CAD --timeframe 1h

For each exchange it asks for candles starting in 2015 and reports the first date it gets back,
plus whether the API can page through the whole history (needed by data.fetch).
"""
import argparse

import pandas as pd

EXCHANGES = ["coinbase", "kraken", "ndax", "bitstamp", "bitfinex"]
LIMITS = {"coinbase": 300, "kraken": 720}


def check(name, symbol, timeframe):
    import ccxt

    ex = getattr(ccxt, name)({"enableRateLimit": True})
    ex.load_markets()
    if symbol not in ex.markets:
        return "pair not listed"
    limit = LIMITS.get(name, 500)
    # Some APIs return nothing when the requested window is before the pair existed, so try
    # each year from 2015 until one returns candles, then narrow down month by month.
    now_year = pd.Timestamp.now("UTC").year
    first, year = None, None
    for y in range(2015, now_year + 1):
        batch = ex.fetch_ohlcv(symbol, timeframe, since=ex.parse8601(f"{y}-01-01T00:00:00Z"), limit=limit)
        if batch:
            first, year = batch, y
            break
    if not first:
        return "listed, but no candles returned"
    if year > 2015:   # data starts sometime in the previous year? check its months
        for m in range(1, 13):
            batch = ex.fetch_ohlcv(symbol, timeframe, limit=limit,
                                   since=ex.parse8601(f"{year - 1}-{m:02d}-01T00:00:00Z"))
            if batch:
                first = batch
                break
    start = pd.to_datetime(first[0][0], unit="ms")
    years = (pd.Timestamp.now("UTC").tz_localize(None) - start).days / 365
    note = ""
    if name == "kraken" and years < 0.5:
        note = "  (Kraken's API only serves the last 720 candles - full history needs their CSV downloads)"
    return f"history from {start:%Y-%m-%d}  (~{years:.1f} years available via API){note}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("symbol", nargs="?", default="BTC/CAD")
    p.add_argument("--timeframe", default="15m")
    args = p.parse_args()
    print(f"Checking {args.symbol} {args.timeframe} candles...\n")
    for name in EXCHANGES:
        try:
            result = check(name, args.symbol, args.timeframe)
        except Exception as e:
            result = f"error ({type(e).__name__}: {str(e)[:80]})"
        print(f"  {name:<10} {result}")
    print("\nAnything with several years of history can be used as data_exchange in config.yaml.")


if __name__ == "__main__":
    main()
