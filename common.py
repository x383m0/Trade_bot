"""Shared helpers: config, data loading, features, chronological split, normalization, metrics."""
import json

import numpy as np
import pandas as pd
import yaml

from env.features import make_features, warmup_bars
from env.trading_env import TradingEnv, position_context


def pair_of(symbol):
    return symbol.replace("/", "")


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data_exchange", cfg["exchange"])
    cfg["live_symbol"] = cfg.get("live_symbol") or cfg["symbol"]   # pair used for paper/live trading
    # the coin we trade is always part of training
    train = list(dict.fromkeys(cfg.get("train_symbols") or [cfg["symbol"]]))
    if cfg["symbol"] not in train:
        train.insert(0, cfg["symbol"])
    cfg["train_symbols"] = train
    keys = {"timeframe": cfg["timeframe"], "pair": pair_of(cfg["symbol"]),
            "exchange": cfg["exchange"], "data_exchange": cfg["data_exchange"]}
    cfg["_data_template"] = cfg["paths"]["data"]
    cfg["paths"] = {k: v.format(**keys) for k, v in cfg["paths"].items()}
    return cfg


def data_path(cfg, symbol):
    return cfg["_data_template"].format(timeframe=cfg["timeframe"], pair=pair_of(symbol),
                                        exchange=cfg["exchange"], data_exchange=cfg["data_exchange"])


def load_splits(cfg):
    """Returns ({symbol: {train|val|test: (ohlcv_df, normalized_features_df)}}, norm_stats).

    All coins are split on the SAME dates (taken from the traded coin), so no coin's training
    data overlaps the traded coin's validation/test period - otherwise correlated coins would
    leak the future into training.
    Each coin is normalized with its OWN train-period stats (so a volatile coin like SOL looks
    comparable to BTC), and those stats are saved for live trading.
    """
    tf = cfg["timeframe"]
    raw = {}
    for sym in cfg["train_symbols"]:
        try:
            df = pd.read_parquet(data_path(cfg, sym))
        except FileNotFoundError:
            if sym == cfg["symbol"]:
                raise SystemExit(f"{data_path(cfg, sym)} not found - run `python -m data.fetch`.")
            print(f"warning: no data for {sym} ({data_path(cfg, sym)}) - skipping it. "
                  "Run `python -m data.fetch` to download it.")
            continue
        feats = make_features(df, tf)
        ok = feats.notna().all(axis=1)
        raw[sym] = (df[ok].reset_index(drop=True), feats[ok].reset_index(drop=True))

    # split dates from the traded coin
    ts = raw[cfg["symbol"]][0]["ts"]
    n = len(ts)
    cut_val = int(ts.iloc[int(n * cfg["split"]["train"])])
    cut_test = int(ts.iloc[int(n * (cfg["split"]["train"] + cfg["split"]["val"]))])

    splits, norm = {}, {"columns": None, "symbols": {}, "val_start_ts": cut_val,
                        "test_start_ts": cut_test, "timeframe": tf}
    for sym, (df, feats) in raw.items():
        parts = {"train": df["ts"] < cut_val,
                 "val": (df["ts"] >= cut_val) & (df["ts"] < cut_test),
                 "test": df["ts"] >= cut_test}
        tr = feats[parts["train"]]
        if len(tr) < 1000:
            print(f"warning: {sym} has only {len(tr)} training bars - skipping it.")
            continue
        mu, sd = tr.mean(), tr.std() + 1e-8
        norm["columns"] = list(feats.columns)
        norm["symbols"][sym] = {"mean": mu.tolist(), "std": sd.tolist()}
        nf = (feats - mu) / sd
        splits[sym] = {k: (df[m].reset_index(drop=True), nf[m].reset_index(drop=True))
                       for k, m in parts.items()}
    return splits, norm


def norm_path(cfg):
    return cfg["paths"]["model"] + "_norm.json"


def save_norm(cfg, norm):
    with open(norm_path(cfg), "w") as f:
        json.dump(norm, f, indent=1)


def load_norm(cfg):
    try:
        with open(norm_path(cfg)) as f:
            norm = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{norm_path(cfg)} not found - run `python train.py` first.")
    if "symbols" not in norm:
        raise SystemExit("Model is from an older version (before multi-coin features) - "
                         "run `python train.py` again.")
    return norm


def required_history(cfg):
    """Closed candles needed to build one live observation."""
    return warmup_bars(cfg["timeframe"]) + cfg["train"]["window"] + 20


def live_observation(bars: pd.DataFrame, norm, symbol, window, pos, upnl_log=0.0, held=0):
    """Build the same observation the env gives at the LAST bar of `bars`."""
    st = norm["symbols"][symbol]
    feats = make_features(bars, norm["timeframe"])[norm["columns"]]
    feats = (feats - np.array(st["mean"])) / np.array(st["std"])
    last = feats.iloc[-window:]
    if len(last) < window or last.isna().any().any():
        return None
    extra = position_context(pos, upnl_log, held)
    obs = np.concatenate([last.values.astype(np.float32).ravel(), extra])
    return np.clip(obs, -10, 10).astype(np.float32)


def make_env(cfg, df, feats, episode_len=None, seed=None):
    """df/feats can be single DataFrames or lists of them (several coins)."""
    if isinstance(df, (list, tuple)):
        close, fv = [d["close"].values for d in df], [f.values for f in feats]
    else:
        close, fv = df["close"].values, feats.values
    return TradingEnv(close, fv, window=cfg["train"]["window"], fee=cfg["fee"],
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
