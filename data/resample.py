"""Build bigger candles (e.g. 4h) from the 1h candles you already downloaded. No internet needed.

Usage:
  python -m data.resample                 # every coin in train_symbols -> the timeframe in config.yaml
  python -m data.resample --to 4h --symbol BTC/USD ETH/USD

Coinbase (where the history comes from) has no 4h candles, so 4h is made from 1h: 4 hourly candles
-> one 4h candle (open of the first, highest high, lowest low, close of the last, volume summed).
Candles start at 00:00, 04:00, 08:00 ... UTC, the same as Kraken's live 4h candles.
"""
import argparse
from pathlib import Path

import pandas as pd

from common import data_path, load_config


def resample_ohlcv(df, timeframe):
    """1h (or smaller) OHLCV -> `timeframe` OHLCV. The last candle is dropped if it isn't finished."""
    rule = timeframe.replace("m", "min") if timeframe.endswith("m") else timeframe
    t = pd.to_datetime(df["ts"], unit="ms")
    kw = {} if rule.endswith("d") else {"origin": "epoch"}
    g = df.set_index(t).resample(rule.upper() if rule.endswith("d") else rule, label="left", closed="left", **kw)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()}).dropna()
    step = pd.Timedelta(rule.upper() if rule.endswith("d") else rule)
    src_step = t.diff().median()
    if len(out) and out.index[-1] + step > t.iloc[-1] + src_step:     # last candle still forming
        out = out.iloc[:-1]
    out["ts"] = (out.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    return out.reset_index(drop=True)[["ts", "open", "high", "low", "close", "volume"]]


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--to", default=cfg["timeframe"], help="target candle size (default: config timeframe)")
    p.add_argument("--src", default="1h", help="candles to build from (default 1h)")
    p.add_argument("--symbol", nargs="+", metavar="COIN", help="coins (default: all train_symbols)")
    args = p.parse_args()
    if args.to == args.src:
        raise SystemExit(f"timeframe is already {args.src} - set `timeframe: 4h` in config.yaml or use --to 4h")
    for sym in args.symbol or cfg["train_symbols"]:
        src = Path(data_path({**cfg, "timeframe": args.src}, sym))
        if not src.exists():
            print(f"{sym}: no {args.src} data ({src}) - download it first: python -m data.fetch --symbol {sym} "
                  f"(with timeframe: {args.src})")
            continue
        out = resample_ohlcv(pd.read_parquet(src), args.to)
        dst = Path(data_path({**cfg, "timeframe": args.to}, sym))
        out.to_parquet(dst)
        print(f"{sym}: {len(out)} {args.to} candles ({pd.to_datetime(out.ts.iloc[0], unit='ms').date()} -> "
              f"{pd.to_datetime(out.ts.iloc[-1], unit='ms').date()}) -> {dst}")


if __name__ == "__main__":
    main()
