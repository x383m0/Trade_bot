"""Gymnasium environment: long-or-flat spot trading.

Action: 0 = flat (hold cash), 1 = long (hold the coin).
Observation: last `window` bars of normalized market features, plus position context:
  [position (0/1), unrealized P&L of the open trade, how long the trade has been open].
Decision at bar t is made on info up to close[t]; the return earned is close[t] -> close[t+1].

Trading rules (all optional, see config.yaml):
  order_type "market": position changes at close[t], paying fee + slippage.
  order_type "limit":  a limit order is placed at close[t] and only fills if the next candle
                       trades through that price (buy: low <= price, sell: high >= price).
                       Pays the cheaper maker fee, but unfilled orders mean missed moves.
  min_hold:  the bot can't sell until it has held for this many candles.
  stop_loss: if the open trade is down this much, it's sold at market no matter what.
Reward types:
  "profit": the portfolio's log-return (minus costs).
  "excess": log-return MINUS buy & hold's, so the bot only scores by beating holding
            (stepping aside in a crash = positive, holding through it = nothing gained).
  drawdown_penalty: extra penalty whenever equity sinks further below its peak.

Training can use several coins: pass lists; each episode picks one at random.
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


def apply_rules(action, pos, held, upnl_log, min_hold=0, stop_loss=0.0):
    """Shared by the env and paper trading. Returns (action, forced_exit)."""
    if pos == 1 and stop_loss > 0 and upnl_log <= np.log(1 - stop_loss):
        return 0, True                      # stop-loss: sell at market
    if pos == 1 and action == 0 and held < min_hold:
        return 1, False                     # too early to sell
    return int(action), False


def limit_filled(side_buy, limit_price, next_high, next_low, margin=0.0002):
    """A limit order at `limit_price` fills only if the next candle trades THROUGH it by
    `margin` (just touching isn't enough: other orders at that price are ahead in the queue)."""
    if side_buy:
        return next_low <= limit_price * (1 - margin)
    return next_high >= limit_price * (1 + margin)


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, close, feats, high=None, low=None, window=16, fee=0.001, slippage=0.0005,
                 trade_penalty=0.0002, episode_len=None, seed=None, order_type="market",
                 maker_fee=0.0025, fill_margin=0.0002, min_hold=0, stop_loss=0.0,
                 reward_type="profit", drawdown_penalty=0.0):
        super().__init__()
        if not isinstance(close, (list, tuple)):          # single coin -> lists of one
            close, feats, high, low = [close], [feats], [high], [low]
        high = high if high is not None else [None] * len(close)
        low = low if low is not None else [None] * len(close)
        self.datasets = []
        for c, f, h, l in zip(close, feats, high, low):
            c = np.asarray(c, np.float64)
            self.datasets.append((c, np.asarray(f, np.float32),
                                  c if h is None else np.asarray(h, np.float64),
                                  c if l is None else np.asarray(l, np.float64)))
        for c, f, _, _ in self.datasets:
            assert len(c) == len(f)
        lens = np.array([len(d[0]) for d in self.datasets], dtype=float)
        self.weights = lens / lens.sum()                  # pick coins by amount of data
        self.window = window
        self.market_cost = fee + slippage
        self.maker_fee, self.fill_margin = maker_fee, fill_margin
        self.order_type = order_type
        self.min_hold, self.stop_loss = int(min_hold), float(stop_loss)
        self.reward_type, self.drawdown_penalty = reward_type, float(drawdown_penalty)
        self.trade_penalty = trade_penalty
        self.episode_len = episode_len  # None = run through the whole slice (evaluation)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(2)
        n_feat = self.datasets[0][1].shape[1]
        self.observation_space = spaces.Box(-10, 10, shape=(window * n_feat + N_EXTRA,),
                                            dtype=np.float32)
        self.close, self.feats, self.high, self.low = self.datasets[0]

    def _upnl(self):
        return np.log(self.close[self.t] / self.entry) if self.pos else 0.0

    def _obs(self):
        w = self.feats[self.t - self.window + 1: self.t + 1].ravel()
        extra = position_context(self.pos, self._upnl(), self.held)
        return np.clip(np.concatenate([w, extra]), -10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        i = int(self.rng.choice(len(self.datasets), p=self.weights)) if self.episode_len else 0
        self.close, self.feats, self.high, self.low = self.datasets[i]
        last_start = len(self.close) - 2
        lo = self.window - 1
        if self.episode_len:
            hi = max(lo, last_start - self.episode_len)
            self.t = int(self.rng.integers(lo, hi + 1))
            self.end = min(self.t + self.episode_len, len(self.close) - 2)
        else:
            self.t = lo
            self.end = len(self.close) - 2
        self.pos, self.entry, self.held = 0, 0.0, 0
        self.equity, self.peak, self.prev_dd = 1.0, 1.0, 0.0
        self.n_trades = 0
        return self._obs(), {}

    def step(self, action):
        t, c = self.t, self.close
        want, forced = apply_rules(int(action), self.pos, self.held, self._upnl(),
                                   self.min_hold, self.stop_loss)
        filled = want != self.pos
        if filled and not forced and self.order_type == "limit":
            filled = limit_filled(want == 1, c[t], self.high[t + 1], self.low[t + 1], self.fill_margin)
        new_pos = want if filled else self.pos
        trade = abs(new_pos - self.pos)
        cost = self.market_cost if (forced or self.order_type == "market") else self.maker_fee
        if new_pos == 1 and self.pos == 0:
            self.entry, self.held = c[t], 0

        r_mkt = np.log(c[t + 1] / c[t])
        net = r_mkt * new_pos - trade * cost
        self.equity *= np.exp(net)
        reward = (net - r_mkt) if self.reward_type == "excess" else net
        reward -= trade * self.trade_penalty
        if self.drawdown_penalty:
            self.peak = max(self.peak, self.equity)
            dd = 1 - self.equity / self.peak
            reward -= self.drawdown_penalty * max(0.0, dd - self.prev_dd)
            self.prev_dd = dd

        self.n_trades += int(trade > 0)
        self.pos = new_pos
        self.t += 1
        self.held = self.held + 1 if self.pos else 0
        at_end = self.t >= self.end
        # Training episodes (episode_len set) are just a random slice of history: reaching the end
        # is a TIME LIMIT, not the end of the world, so report it as "truncated" - PPO then keeps
        # valuing what comes after. Evaluation runs (whole period) really end: "terminated".
        terminated = at_end and not self.episode_len
        truncated = at_end and bool(self.episode_len)
        return self._obs(), float(reward * 100), terminated, truncated, {
            "equity": self.equity, "n_trades": self.n_trades, "pos": self.pos, "forced": forced}
