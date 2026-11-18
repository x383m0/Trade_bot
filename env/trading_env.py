"""Gymnasium environment: long-or-flat spot trading.

Action: 0 = flat (hold cash), 1 = long (hold the coin).
Observation: last `window` bars of normalized market features, plus position context:
  [position (0/1), unrealized P&L of the open trade, how long the trade has been open].
Reward: log-return of the portfolio each bar, minus fees/slippage when the position changes,
minus a small extra penalty per trade.
Decision at bar t is made on info up to close[t]; the return earned is close[t] -> close[t+1].

Training can use several coins: pass lists of (close, feats); each episode picks one at random.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_EXTRA = 3         # position context values appended after the market features
POS_INDEX = -3      # where the position (0/1) sits in the observation
UPNL_SCALE = 20.0   # unrealized log-return * 20 -> ~±1 for a ±5% trade
HELD_SCALE_BARS = 96


def position_context(pos, upnl_log, held_bars):
    return [float(pos), float(upnl_log) * UPNL_SCALE, min(held_bars / HELD_SCALE_BARS, 3.0)]


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, close, feats, window=16, fee=0.001, slippage=0.0005,
                 trade_penalty=0.0002, episode_len=None, seed=None):
        super().__init__()
        if isinstance(close, (list, tuple)):          # several coins
            self.datasets = [(np.asarray(c, np.float64), np.asarray(f, np.float32))
                             for c, f in zip(close, feats)]
        else:
            self.datasets = [(np.asarray(close, np.float64), np.asarray(feats, np.float32))]
        for c, f in self.datasets:
            assert len(c) == len(f)
        # pick coins in proportion to how much data they have
        lens = np.array([len(c) for c, _ in self.datasets], dtype=float)
        self.weights = lens / lens.sum()
        self.window = window
        self.cost = fee + slippage
        self.trade_penalty = trade_penalty
        self.episode_len = episode_len  # None = run through the whole slice (evaluation)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(2)
        n_feat = self.datasets[0][1].shape[1]
        self.observation_space = spaces.Box(-10, 10, shape=(window * n_feat + N_EXTRA,),
                                            dtype=np.float32)
        self.close, self.feats = self.datasets[0]

    def _obs(self):
        w = self.feats[self.t - self.window + 1: self.t + 1].ravel()
        upnl = np.log(self.close[self.t] / self.entry) if self.pos else 0.0
        extra = position_context(self.pos, upnl, self.held)
        return np.clip(np.concatenate([w, extra]), -10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        i = int(self.rng.choice(len(self.datasets), p=self.weights)) if self.episode_len else 0
        self.close, self.feats = self.datasets[i]
        last_start = len(self.close) - 2
        lo = self.window - 1
        if self.episode_len:
            hi = max(lo, last_start - self.episode_len)
            self.t = int(self.rng.integers(lo, hi + 1))
            self.end = min(self.t + self.episode_len, len(self.close) - 2)
        else:
            self.t = lo
            self.end = len(self.close) - 2
        self.pos, self.entry, self.held = 0.0, 0.0, 0
        self.equity = 1.0
        self.n_trades = 0
        return self._obs(), {}

    def step(self, action):
        new_pos = float(action)
        trade = abs(new_pos - self.pos)
        if new_pos == 1 and self.pos == 0:
            self.entry, self.held = self.close[self.t], 0
        r = np.log(self.close[self.t + 1] / self.close[self.t]) * new_pos
        net = r - trade * self.cost
        reward = net - trade * self.trade_penalty
        self.equity *= np.exp(net)
        self.n_trades += int(trade > 0)
        self.pos = new_pos
        self.t += 1
        self.held = self.held + 1 if self.pos else 0
        terminated = self.t >= self.end
        return self._obs(), float(reward * 100), terminated, False, {
            "equity": self.equity, "n_trades": self.n_trades}
