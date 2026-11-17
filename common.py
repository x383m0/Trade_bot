"""Shared helpers: load data, build features, chronological split, normalization."""
import numpy as np
import pandas as pd
import yaml

from env.features import make_features
from env.trading_env import TradingEnv


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_splits(cfg):
    df = pd.read_parquet(cfg["paths"]["data"])
    feats = make_features(df)
    ok = feats.notna().all(axis=1)
    df, feats = df[ok].reset_index(drop=True), feats[ok].reset_index(drop=True)

    n = len(df)
    a = int(n * cfg["split"]["train"])
    b = int(n * (cfg["split"]["train"] + cfg["split"]["val"]))
    parts = {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}

    # normalize with TRAIN statistics only (no look-ahead leakage)
    mu, sd = feats.iloc[:a].mean(), feats.iloc[:a].std() + 1e-8
    feats = (feats - mu) / sd

    return {k: (df.iloc[s].reset_index(drop=True), feats.iloc[s].reset_index(drop=True))
            for k, s in parts.items()}


def make_env(cfg, df, feats, episode_len=None, seed=None):
    return TradingEnv(df["close"].values, feats.values,
                      window=cfg["train"]["window"], fee=cfg["fee"],
                      slippage=cfg["slippage"], trade_penalty=cfg["trade_penalty"],
                      episode_len=episode_len, seed=seed)


def bars_per_year(timeframe):
    unit = {"m": 1, "h": 60, "d": 1440}[timeframe[-1]]
    return 365 * 24 * 60 / (int(timeframe[:-1]) * unit)


def metrics(equity, timeframe):
    equity = np.asarray(equity)
    rets = np.diff(np.log(equity))
    bpy = bars_per_year(timeframe)
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(bpy)
    peak = np.maximum.accumulate(equity)
    return {"total_return_%": (equity[-1] - 1) * 100,
            "sharpe": sharpe,
            "max_drawdown_%": ((equity / peak).min() - 1) * 100}
