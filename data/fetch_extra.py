"""Download information that price candles don't contain (free, no API key):

  Fear & Greed Index  (alternative.me)  daily crowd sentiment, 0 = extreme fear, 100 = extreme greed
  Funding rates       (Kraken Futures)  what futures traders pay to bet on BTC going up.
                                        High = crowded bullish bets (often before drops)

Usage:
  python -m data.fetch_extra            # saves data/fear_greed.parquet and data/funding_btc.parquet

Both are saved as one value per UTC day. Run it again any time to update.
"""
import time
from pathlib import Path

import pandas as pd
import requests

FGI_URL = "https://api.alternative.me/fng/?limit=0&format=json"
FUNDING_URL = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol={sym}"
FUNDING_SYMBOLS = ["PI_XBTUSD", "PF_XBTUSD"]   # inverse perpetual (older) + linear perpetual (newer)
OUT = Path("data")


def get_json(url, tries=4):
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "trade-bot-research"})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt * 3
            print(f"  {type(e).__name__}: {e} - retrying in {wait}s")
            time.sleep(wait)


def parse_fear_greed(js):
    """alternative.me -> DataFrame(date, fear_greed) one row per UTC day."""
    rows = js.get("data") or []
    if not rows:
        raise ValueError(f"unexpected Fear & Greed response: {str(js)[:300]}")
    df = pd.DataFrame({"date": pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s").normalize(),
                       "fear_greed": [float(r["value"]) for r in rows]})
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def parse_funding(js, symbol):
    """Kraken Futures -> DataFrame(time, rate). Uses the RELATIVE rate (a fraction of the position
    per funding period) so the two contract types are comparable."""
    rows = js.get("rates")
    if rows is None:
        raise ValueError(f"unexpected funding response for {symbol}: {str(js)[:300]}")
    if not rows:
        return pd.DataFrame(columns=["time", "rate"])
    key = "relativeFundingRate" if "relativeFundingRate" in rows[0] else "fundingRate"
    return pd.DataFrame({"time": pd.to_datetime([r["timestamp"] for r in rows], utc=True).tz_localize(None),
                         "rate": [float(r[key]) for r in rows]})


def daily_funding(frames):
    """Average funding per UTC day, across the contracts available that day."""
    allr = pd.concat([f for f in frames if len(f)], ignore_index=True)
    allr["date"] = allr["time"].dt.normalize()
    return allr.groupby("date")["rate"].mean().rename("funding").reset_index()


def main():
    OUT.mkdir(exist_ok=True)

    print("Fear & Greed Index (alternative.me)...")
    fgi = parse_fear_greed(get_json(FGI_URL))
    fgi.to_parquet(OUT / "fear_greed.parquet")
    print(f"  saved {len(fgi)} days ({fgi.date.iloc[0].date()} -> {fgi.date.iloc[-1].date()}) "
          f"-> {OUT / 'fear_greed.parquet'}")

    print("Funding rates (Kraken Futures)...")
    frames = []
    for sym in FUNDING_SYMBOLS:
        try:
            f = parse_funding(get_json(FUNDING_URL.format(sym=sym)), sym)
            print(f"  {sym}: {len(f)} funding payments"
                  + (f" ({f.time.min().date()} -> {f.time.max().date()})" if len(f) else ""))
            frames.append(f)
        except Exception as e:     # one contract failing shouldn't stop the other
            print(f"  {sym}: failed ({type(e).__name__}: {e})")
    if not any(len(f) for f in frames):
        raise SystemExit("no funding data downloaded - paste this output to get help")
    fund = daily_funding(frames)
    fund.to_parquet(OUT / "funding_btc.parquet")
    print(f"  saved {len(fund)} days ({fund.date.iloc[0].date()} -> {fund.date.iloc[-1].date()}) "
          f"-> {OUT / 'funding_btc.parquet'}")


if __name__ == "__main__":
    main()
