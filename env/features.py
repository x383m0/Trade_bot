"""Market features the agent sees at every bar.

All features are stationary (returns, ratios, positions in a range) - never raw prices - so
what the agent learns transfers across years and across coins (BTC, ETH, SOL...).
Horizons are defined in HOURS, so the same features work on 5m, 15m or 1h candles.
"""
import numpy as np
import pandas as pd

LONGEST_HOURS = 96  # longest INTRADAY lookback (4 days) - candles of history live trading needs
LONG_TERM_DAYS = 200  # longest DAILY lookback (long-term trend features, from daily candles)
DAILY_HISTORY = LONG_TERM_DAYS + 10   # completed daily candles live trading needs
# the long-term ("trend") features - the only ones used with train.features: trend_only
LONG_TERM_COLUMNS = ["ret_7d", "ret_30d", "ret_90d", "dist_ma_30d", "dist_ma_200d", "ma30_vs_ma200",
                     "from_high_90d", "vol_30d", "vol_regime_30d"]


def bars_per_hour(timeframe: str) -> float:
    unit = {"m": 1, "h": 60, "d": 1440}[timeframe[-1]]
    return 60 / (int(timeframe[:-1]) * unit)


def warmup_bars(timeframe: str) -> int:
    """Bars of history needed before all features are defined."""
    return int(np.ceil(LONGEST_HOURS * bars_per_hour(timeframe))) + 1


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily candles (UTC days) built from intraday candles. Same shape as an exchange's 1d OHLCV."""
    t = pd.to_datetime(df["ts"], unit="ms").dt.floor("D")
    g = df.groupby(t)
    d = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    d = d.asfreq("D").ffill()          # a day with no candles keeps the previous close
    d["ts"] = (d.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)   # unit-safe (pandas 3)
    return d.reset_index(drop=True)


def long_term_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """The big picture: is this a weeks/months-long uptrend or downtrend?

    Built from DAILY candles, using only days that were already finished before the current
    candle's day (no peeking), compared with the current price. Works live too: Kraken gives
    720 daily candles, while its 720 hourly candles would only cover 30 days.
    """
    d = daily.sort_values("ts").reset_index(drop=True)
    day = pd.to_datetime(d["ts"], unit="ms").dt.floor("D")
    dc = d["close"].astype(float)
    dr = np.log(dc).diff()
    ref = pd.DataFrame({
        "c7": dc.shift(6), "c30": dc.shift(29), "c90": dc.shift(89),
        "ma30": dc.rolling(30).mean(), "ma200": dc.rolling(LONG_TERM_DAYS).mean(),
        "hi90": d["high"].rolling(90).max(), "vol30": dr.rolling(30).std(),
        "vol200": dr.rolling(LONG_TERM_DAYS).std(),
    })
    ref.index = day.values
    # each candle uses the last COMPLETED day (the day before its own)
    prev_day = pd.to_datetime(df["ts"], unit="ms").dt.floor("D") - pd.Timedelta(days=1)
    r = ref.reindex(prev_day.values)
    r.index = df.index
    c = df["close"].astype(float)
    f = pd.DataFrame(index=df.index)
    f["ret_7d"] = np.log(c / r["c7"])
    f["ret_30d"] = np.log(c / r["c30"])
    f["ret_90d"] = np.log(c / r["c90"])
    f["dist_ma_30d"] = c / r["ma30"] - 1
    f["dist_ma_200d"] = c / r["ma200"] - 1
    f["ma30_vs_ma200"] = r["ma30"] / r["ma200"] - 1          # "golden/death cross" regime
    f["from_high_90d"] = c / r["hi90"] - 1                   # drawdown from the 90-day high
    f["vol_30d"] = r["vol30"]
    f["vol_regime_30d"] = np.log(r["vol30"] / (r["vol200"] + 1e-12) + 1e-12)
    return f


def make_features(df: pd.DataFrame, timeframe: str = "15m", long_term: bool = False,
                  daily: pd.DataFrame = None) -> pd.DataFrame:
    """long_term=True adds the daily trend features. `daily` = daily candles; if not given they
    are built from `df` itself (fine for history; live trading passes the exchange's 1d candles)."""
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

    if long_term:
        lt = long_term_features(df, to_daily(df) if daily is None else daily)
        f = pd.concat([f, lt], axis=1)
    return f.replace([np.inf, -np.inf], np.nan)
