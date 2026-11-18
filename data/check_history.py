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
    since = ex.parse8601("2015-01-01T00:00:00Z")
    limit = LIMITS.get(name, 500)
    first = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not first:
        # some APIs return nothing for very old dates: try 2 years back
        since = ex.milliseconds() - 2 * 365 * 86400 * 1000
        first = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not first:
        return "listed, but no candles returned"
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
