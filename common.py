"""Shared helpers: config, data loading, features, chronological split, normalization, metrics."""
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from env.features import make_features, warmup_bars
from env.trading_env import TradingEnv, position_context


# ---- safe file sharing: the dashboard reads files while other scripts are writing them ----
def write_json(path, obj):
    """Write JSON so a reader never sees a half-written file: write a temp file, then swap it in
    (the swap is instant). Windows can refuse the swap while a reader has the file open for a
    split second, so retry a few times."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def read_json(path, default=None, retries=10):
    """Read a JSON file that another script may be rewriting right now (retries briefly)."""
    path = Path(path)
    for _ in range(retries):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(0.05)
    return default


def read_csv(path, **kw):
    """Read a CSV that another script may be appending to right now: only complete lines are
    parsed (a half-written last line is ignored) and an empty/just-created file gives an empty
    table instead of an error."""
    path = Path(path)
    for _ in range(10):
        try:
            text = path.read_text()
            break
        except PermissionError:
            time.sleep(0.05)
    else:
        text = ""
    if "\n" not in text:
        return pd.DataFrame()
    text = text[: text.rfind("\n") + 1]          # drop a half-written last line
    try:
        return pd.read_csv(io.StringIO(text), **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def long_term(cfg):
    """Long-term trend features on? (train.long_term_features in config.yaml)"""
    return bool(cfg["train"].get("long_term_features", False))


def data_path(cfg, symbol):
    return cfg["_data_template"].format(timeframe=cfg["timeframe"], pair=pair_of(symbol),
                                        exchange=cfg["exchange"], data_exchange=cfg["data_exchange"])


GAP_DAYS = 2   # a hole longer than this in a coin's history splits it into separate pieces


def split_at_gaps(df, max_gap_days=GAP_DAYS):
    """Row positions of each unbroken piece of history. Some coins have holes: e.g. Coinbase
    delisted XRP from Jan 2021 to mid-2023. Joining across a hole would look like one candle with a
    2-year price move, and indicators would mix data from both sides."""
    ts = df["ts"].values
    cut = np.where(np.diff(ts) > max_gap_days * 86_400_000)[0] + 1
    return np.split(np.arange(len(df)), cut)


def pieces(df, feats, min_len):
    """[(df, feats)] for each unbroken piece with at least `min_len` candles."""
    return [(df.iloc[i].reset_index(drop=True), feats.iloc[i].reset_index(drop=True))
            for i in split_at_gaps(df) if len(i) >= min_len]


def load_splits(cfg, fractions=None):
    """Returns ({symbol: {train|val|test: (ohlcv_df, normalized_features_df)}}, norm_stats).

    fractions = (val_start, test_start, test_end) as fractions of the traded coin's history.
    Default: from config.yaml `split` (e.g. 0.70, 0.85, 1.0). Walk-forward passes its own.

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
        segs = split_at_gaps(df)
        if len(segs) > 1:
            gaps = [f"{pd.to_datetime(df['ts'].iloc[a[-1]], unit='ms').date()} -> "
                    f"{pd.to_datetime(df['ts'].iloc[b[0]], unit='ms').date()}" for a, b in zip(segs, segs[1:])]
            print(f"note: {sym} has holes in its history ({', '.join(gaps)}) - each side is used separately")
        # indicators are computed on each unbroken piece (the first days after a hole get dropped below,
        # until there is enough history again)
        feats = pd.concat([make_features(df.iloc[i].reset_index(drop=True), tf, long_term=long_term(cfg))
                           for i in segs], ignore_index=True)
        ok = feats.notna().all(axis=1)
        raw[sym] = (df[ok].reset_index(drop=True), feats[ok].reset_index(drop=True))

    # split dates from the traded coin
    ts = raw[cfg["symbol"]][0]["ts"]
    n = len(ts)
    if fractions is None:
        fractions = (cfg["split"]["train"], cfg["split"]["train"] + cfg["split"]["val"], 1.0)
    at = lambda frac: int(ts.iloc[min(n - 1, int(n * frac))])
    cut_val, cut_test = at(fractions[0]), at(fractions[1])
    cut_end = at(fractions[2]) if fractions[2] < 1.0 else int(ts.iloc[-1]) + 1

    splits, norm = {}, {"columns": None, "symbols": {}, "val_start_ts": cut_val,
                        "test_start_ts": cut_test, "timeframe": tf, "long_term": long_term(cfg)}
    for sym, (df, feats) in raw.items():
        parts = {"train": df["ts"] < cut_val,
                 "val": (df["ts"] >= cut_val) & (df["ts"] < cut_test),
                 "test": (df["ts"] >= cut_test) & (df["ts"] < cut_end)}
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


def live_observation(bars: pd.DataFrame, norm, symbol, window, pos, upnl_log=0.0, held=0,
                     daily: pd.DataFrame = None):
    """Build the same observation the env gives at the LAST bar of `bars`.
    `daily` = daily candles (needed when the model uses long-term features)."""
    st = norm["symbols"][symbol]
    lt = norm.get("long_term", False)        # follow the MODEL's setting, not the current config
    if lt and daily is None:
        raise ValueError("this model uses long-term features: pass daily candles")
    feats = make_features(bars, norm["timeframe"], long_term=lt, daily=daily)[norm["columns"]]
    feats = (feats - np.array(st["mean"])) / np.array(st["std"])
    last = feats.iloc[-window:]
    if len(last) < window or last.isna().any().any():
        return None
    extra = position_context(pos, upnl_log, held)
    obs = np.concatenate([last.values.astype(np.float32).ravel(), extra])
    return np.clip(obs, -10, 10).astype(np.float32)


def trade_rules(cfg):
    """Order type, holding/stop rules and reward settings from config.yaml (all optional)."""
    o, r, w = cfg.get("orders") or {}, cfg.get("rules") or {}, cfg.get("reward") or {}
    return {"order_type": o.get("type", "market"), "maker_fee": float(o.get("maker_fee", 0.0025)),
            "fill_margin": float(o.get("fill_margin", 0.0002)),
            "min_hold": int(r.get("min_hold", 0) or 0), "stop_loss": float(r.get("stop_loss", 0) or 0),
            "reward_type": w.get("type", "profit"),
            "drawdown_penalty": float(w.get("drawdown_penalty", 0) or 0)}


def trade_cost(cfg):
    """Cost per position change: maker fee for limit orders, fee + slippage for market."""
    tr = trade_rules(cfg)
    return tr["maker_fee"] if tr["order_type"] == "limit" else cfg["fee"] + cfg["slippage"]


def make_env(cfg, df, feats, episode_len=None, seed=None):
    """df/feats can be single DataFrames or lists of them (several coins)."""
    dfs, fs = (df, feats) if isinstance(df, (list, tuple)) else ([df], [feats])
    return TradingEnv([d["close"].values for d in dfs], [f.values for f in fs],
                      high=[d["high"].values for d in dfs], low=[d["low"].values for d in dfs],
                      window=cfg["train"]["window"], fee=cfg["fee"], slippage=cfg["slippage"],
                      trade_penalty=cfg["trade_penalty"], episode_len=episode_len, seed=seed,
                      **trade_rules(cfg))


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
