import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary features only (returns/ratios), no raw prices, to limit overfitting."""
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["ret1"] = np.log(c).diff()
    f["ret4"] = np.log(c).diff(4)
    f["ret16"] = np.log(c).diff(16)
    f["vol16"] = f["ret1"].rolling(16).std()
    f["vol96"] = f["ret1"].rolling(96).std()
    f["hl_range"] = (df["high"] - df["low"]) / c
    f["vol_ratio"] = df["volume"] / df["volume"].rolling(96).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = (100 - 100 / (1 + gain / (loss + 1e-12))) / 100 - 0.5

    ma = c.rolling(48).mean()
    f["dist_ma48"] = c / ma - 1

    t = pd.to_datetime(df["ts"], unit="ms")
    hour = t.dt.hour + t.dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    f = f.replace([np.inf, -np.inf], np.nan)
    return f
