"""Load the trading agent: a single PPO model, or an ensemble of all trained seeds that votes.

Ensemble rule: each model votes flat (0) or long (1). The bot only CHANGES position when
more than `min_agree` of the models vote for the new position; otherwise it keeps what it has.
With 3 models and min_agree 0.5 that's a 2-of-3 majority. Disagreement -> stay put -> fewer
trades when the agents are unsure.
"""
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


def ensemble_path(cfg):
    return cfg["paths"]["model"] + "_ensemble.json"


class Ensemble:
    def __init__(self, models, names, min_agree=0.5):
        self.models, self.names, self.min_agree = models, names, min_agree
        self.last_votes = None

    def predict(self, obs, deterministic=True):
        votes = np.array([int(m.predict(obs, deterministic=deterministic)[0]) for m in self.models])
        self.last_votes = votes
        pos = int(round(float(obs[-1])))        # current position is the last obs value
        share_long = votes.mean()
        if pos == 0 and share_long > self.min_agree:
            action = 1
        elif pos == 1 and (1 - share_long) > self.min_agree:
            action = 0
        else:
            action = pos
        return np.array(action), None

    def votes_str(self):
        if self.last_votes is None:
            return ""
        return f"{int(self.last_votes.sum())}/{len(self.last_votes)} long"

    def __len__(self):
        return len(self.models)


class Single:
    """Same interface as Ensemble for a single model."""

    def __init__(self, model, name="single"):
        self.model, self.names, self.models = model, [name], [model]
        self.last_votes = None

    def predict(self, obs, deterministic=True):
        a, s = self.model.predict(obs, deterministic=deterministic)
        self.last_votes = np.array([int(a)])
        return a, s

    def votes_str(self):
        return ""

    def __len__(self):
        return 1


def load_agent(cfg, verbose=True):
    """Ensemble of all seeds from the last training run if `ensemble: true` and available,
    otherwise the single best model."""
    ens_cfg = cfg.get("ensemble", {}) or {}
    manifest = Path(ensemble_path(cfg))
    if ens_cfg.get("enabled", True) and manifest.exists():
        m = json.loads(manifest.read_text())
        paths = [p for p in m["models"] if Path(p + ".zip").exists()]
        if len(paths) > 1:
            agent = Ensemble([PPO.load(p) for p in paths], [f"seed {s}" for s in m["seeds"]],
                             ens_cfg.get("min_agree", 0.5))
            if verbose:
                print(f"agent: ensemble of {len(agent)} seeds (majority vote)")
            return agent
    if verbose:
        print("agent: single best model")
    return Single(PPO.load(cfg["paths"]["model"]), "best seed")


def load_seed_models(cfg):
    """Each seed's model separately (for comparing them in the backtest)."""
    manifest = Path(ensemble_path(cfg))
    if not manifest.exists():
        return {}
    m = json.loads(manifest.read_text())
    return {f"seed {s}": Single(PPO.load(p), f"seed {s}")
            for s, p in zip(m["seeds"], m["models"]) if Path(p + ".zip").exists()}
