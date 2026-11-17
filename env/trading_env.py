"""Gymnasium environment: long-or-flat spot trading on one asset.

Action: 0 = flat (hold cash), 1 = long (hold the asset).
Observation: last `window` bars of normalized features + current position.
Reward: log-return of the portfolio each bar, minus fees/slippage when the
position changes, minus a small extra penalty per trade.
Decision at bar t is made on info up to close[t]; the return earned is close[t] -> close[t+1].
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, close, feats, window=32, fee=0.001, slippage=0.0005,
                 trade_penalty=0.0002, episode_len=None, seed=None):
        super().__init__()
        assert len(close) == len(feats)
        self.close = np.asarray(close, dtype=np.float64)
        self.feats = np.asarray(feats, dtype=np.float32)
        self.window = window
        self.cost = fee + slippage
        self.trade_penalty = trade_penalty
        self.episode_len = episode_len  # None = run through whole slice (eval)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(2)
        n_feat = self.feats.shape[1]
        self.observation_space = spaces.Box(-10, 10, shape=(window * n_feat + 1,), dtype=np.float32)

    def _obs(self):
        w = self.feats[self.t - self.window + 1: self.t + 1].ravel()
        return np.clip(np.append(w, self.pos), -10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        last_start = len(self.close) - 2
        lo = self.window - 1
        if self.episode_len:
            hi = max(lo, last_start - self.episode_len)
            self.t = int(self.rng.integers(lo, hi + 1))
            self.end = min(self.t + self.episode_len, len(self.close) - 2)
        else:
            self.t = lo
            self.end = len(self.close) - 2
        self.pos = 0.0
        self.equity = 1.0
        self.n_trades = 0
        return self._obs(), {}

    def step(self, action):
        new_pos = float(action)
        trade = abs(new_pos - self.pos)
        r = np.log(self.close[self.t + 1] / self.close[self.t]) * new_pos
        cost = trade * self.cost
        net = r - cost
        reward = net - trade * self.trade_penalty
        self.equity *= np.exp(net)
        self.n_trades += int(trade > 0)
        self.pos = new_pos
        self.t += 1
        terminated = self.t >= self.end
        return self._obs(), float(reward * 100), terminated, False, {
            "equity": self.equity, "n_trades": self.n_trades}
