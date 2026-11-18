"""Load the trading agent: a single PPO model, or an ensemble of trained seeds.

Ensemble voting (config.yaml -> ensemble.vote):
  "soft" (default): average how CONFIDENT the seeds are that holding is right (each seed's
          probability of "long"). Buy when the average is above 50% + margin, sell when it's
          below 50% - margin. One confident seed can outweigh two unsure ones, and seeds that
          buy at slightly different times still add up instead of cancelling out.
  "majority": each seed votes flat/long; the position only changes when more than
          `min_agree` of them vote for it. With 3 seeds that's 2-of-3. Strict: if the seeds
          buy at different times, the majority rarely agrees and the bot sits in cash.
"""
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from env.trading_env import POS_INDEX


def ensemble_path(cfg):
    return cfg["paths"]["model"] + "_ensemble.json"


def prob_long(model, obs):
    """The model's probability that "long" is the right action for this observation."""
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        probs = model.policy.get_distribution(obs_t).distribution.probs
    return float(probs[0, 1])


class Ensemble:
    def __init__(self, models, names, vote="soft", min_agree=0.5, margin=0.0):
        self.models, self.names = models, names
        self.vote, self.min_agree, self.margin = vote, min_agree, margin
        self.last_votes, self.last_p = None, None

    def predict(self, obs, deterministic=True):
        pos = int(round(float(obs[POS_INDEX])))  # current position, from the observation
        if self.vote == "soft":
            ps = np.array([prob_long(m, obs) for m in self.models])
            self.last_p, self.last_votes = ps, (ps > 0.5).astype(int)
            p = ps.mean()
            if pos == 0 and p > 0.5 + self.margin:
                action = 1
            elif pos == 1 and p < 0.5 - self.margin:
                action = 0
            else:
                action = pos
            return np.array(action), None
        votes = np.array([int(m.predict(obs, deterministic=deterministic)[0]) for m in self.models])
        self.last_votes, self.last_p = votes, None
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
        if self.last_p is not None:
            return f"{self.last_p.mean() * 100:.0f}% sure long"
        return f"{int(self.last_votes.sum())}/{len(self.last_votes)} long"

    def label(self):
        return f"Ensemble ({len(self)} seeds, {self.vote} vote)"

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

    def label(self):
        return "PPO agent"

    def __len__(self):
        return 1


def build_agent(cfg, model_paths, names=None, vote=None):
    """Ensemble (or single model) from a list of model paths, using config.yaml's voting rule."""
    ens = cfg.get("ensemble", {}) or {}
    paths = [p for p in model_paths if Path(str(p) + ".zip").exists()]
    names = names or [Path(p).name for p in paths]
    if len(paths) == 1 or not ens.get("enabled", True):
        return Single(PPO.load(paths[0], device="cpu"), names[0])
    return Ensemble([PPO.load(p, device="cpu") for p in paths], names,
                    vote=vote or ens.get("vote", "soft"), min_agree=ens.get("min_agree", 0.5),
                    margin=ens.get("margin", 0.0))


def load_agent(cfg, verbose=True, vote=None):
    """Ensemble of all seeds from the last training run if `ensemble.enabled` and available,
    otherwise the single best model."""
    ens_cfg = cfg.get("ensemble", {}) or {}
    manifest = Path(ensemble_path(cfg))
    if ens_cfg.get("enabled", True) and manifest.exists():
        m = json.loads(manifest.read_text())
        paths = [p for p in m["models"] if Path(p + ".zip").exists()]
        if len(paths) > 1:
            agent = build_agent(cfg, paths, [f"seed {s}" for s in m["seeds"]], vote=vote)
            if verbose:
                print(f"agent: {agent.label()}")
            return agent
    if verbose:
        print("agent: single best model")
    return Single(PPO.load(cfg["paths"]["model"], device="cpu"), "best seed")


def load_seed_models(cfg):
    """Each seed's model separately (for comparing them in the backtest)."""
    manifest = Path(ensemble_path(cfg))
    if not manifest.exists():
        return {}
    m = json.loads(manifest.read_text())
    return {f"seed {s}": Single(PPO.load(p, device="cpu"), f"seed {s}")
            for s, p in zip(m["seeds"], m["models"]) if Path(p + ".zip").exists()}
