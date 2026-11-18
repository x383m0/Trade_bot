"""Shared helpers: config, data loading, features, chronological split, normalization, metrics."""
import json

import numpy as np
import pandas as pd
import yaml

from env.features import make_features
from env.trading_env import TradingEnv


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data_exchange", cfg["exchange"])
    keys = {"timeframe": cfg["timeframe"], "pair": cfg["symbol"].replace("/", ""),
            "exchange": cfg["exchange"], "data_exchange": cfg["data_exchange"]}
    cfg["paths"] = {k: v.format(**keys) for k, v in cfg["paths"].items()}
    return cfg


def load_splits(cfg):
    """Returns ({train|val|test: (ohlcv_df, normalized_features_df)}, norm_stats)."""
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
    norm = {"columns": list(feats.columns), "mean": mu.tolist(), "std": sd.tolist(),
            "test_start_ts": int(df["ts"].iloc[b])}
    feats = (feats - mu) / sd

    splits = {k: (df.iloc[s].reset_index(drop=True), feats.iloc[s].reset_index(drop=True))
              for k, s in parts.items()}
    return splits, norm


def norm_path(cfg):
    return cfg["paths"]["model"] + "_norm.json"


def save_norm(cfg, norm):
    with open(norm_path(cfg), "w") as f:
        json.dump(norm, f, indent=1)


def load_norm(cfg):
    try:
        with open(norm_path(cfg)) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{norm_path(cfg)} not found - run `python train.py` first.")


def live_observation(bars: pd.DataFrame, norm, window, pos):
    """Build the same observation the env gives at the LAST bar of `bars` (needs ~150+ bars)."""
    feats = make_features(bars)[norm["columns"]]
    feats = (feats - np.array(norm["mean"])) / np.array(norm["std"])
    last = feats.iloc[-window:]
    if len(last) < window or last.isna().any().any():
        return None
    return np.clip(np.append(last.values.astype(np.float32).ravel(), pos), -10, 10).astype(np.float32)


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
