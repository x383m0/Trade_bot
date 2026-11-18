"""Market features the agent sees at every bar.

All features are stationary (returns, ratios, positions in a range) - never raw prices - so
what the agent learns transfers across years and across coins (BTC, ETH, SOL...).
Horizons are defined in HOURS, so the same features work on 5m, 15m or 1h candles.
"""
import numpy as np
import pandas as pd

LONGEST_HOURS = 96  # longest lookback below (4 days) - decides how much history live trading needs


def bars_per_hour(timeframe: str) -> float:
    unit = {"m": 1, "h": 60, "d": 1440}[timeframe[-1]]
    return 60 / (int(timeframe[:-1]) * unit)


def warmup_bars(timeframe: str) -> int:
    """Bars of history needed before all features are defined."""
    return int(np.ceil(LONGEST_HOURS * bars_per_hour(timeframe))) + 1


def make_features(df: pd.DataFrame, timeframe: str = "15m") -> pd.DataFrame:
    bph = bars_per_hour(timeframe)
    n = lambda hours: max(1, int(round(hours * bph)))   # hours -> number of bars
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    logc = np.log(c)
    r1 = logc.diff()
    f = pd.DataFrame(index=df.index)

    # --- momentum: how much price moved over several horizons (the "bigger picture") ---
    f["ret_1bar"] = r1
    for hrs in (1, 4, 24, 96):
        if n(hrs) > 1:
            f[f"ret_{hrs}h"] = logc.diff(n(hrs))

    # --- volatility and its regime (is the market calmer or wilder than usual?) ---
    vol_4h = r1.rolling(max(4, n(4))).std()
    vol_24h = r1.rolling(max(8, n(24))).std()
    f["vol_4h"] = vol_4h
    f["vol_24h"] = vol_24h
    f["vol_regime"] = np.log(vol_4h / (vol_24h + 1e-12) + 1e-12)
    f["bar_range"] = (h - l) / c

    # --- volume spikes ---
    f["volume_spike"] = np.log((v + 1e-9) / (v.rolling(max(8, n(24))).mean() + 1e-9))

    # --- RSI (14 bars) ---
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = (100 - 100 / (1 + gain / (loss + 1e-12))) / 100 - 0.5

    # --- trend: distance from moving averages (1 day, 4 days) ---
    for hrs in (24, 96):
        f[f"dist_ma_{hrs}h"] = c / c.rolling(n(hrs)).mean() - 1

    # --- where is price inside its recent range? (0 = at the low, 1 = at the high) ---
    for hrs in (24, 96):
        hi, lo = h.rolling(n(hrs)).max(), l.rolling(n(hrs)).min()
        f[f"range_pos_{hrs}h"] = (c - lo) / (hi - lo + 1e-12) - 0.5
    f["from_high_96h"] = c / h.rolling(n(96)).max() - 1      # drawdown from the 4-day high

    # --- time: hour of day, day of week (crypto has daily/weekly rhythms) ---
    t = pd.to_datetime(df["ts"], unit="ms")
    hour = t.dt.hour + t.dt.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    dow = t.dt.dayofweek + hour / 24
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    return f.replace([np.inf, -np.inf], np.nan)
