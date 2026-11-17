# Trade_bot — RL crypto trading (learning project)

A PPO agent (Stable-Baselines3) that learns to go **long or flat** on BTC/USDT
using **15-minute** candles, with fees and slippage included in training.

> Learning project. Expect break-even or small losses. Never risk money you can't lose.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

```bash
python -m data.fetch            # download ~2 years of candles (exchange set in config.yaml)
python train.py                 # train PPO, keeps the best checkpoint on the validation split
python backtest.py --split val  # compare vs buy & hold / do nothing, saves reports/backtest_val.png
python backtest.py --split test # ONLY once you're done tuning — the test set is your "unseen future"
```

Offline / pipeline test: `python -m data.fetch --synthetic` generates a random walk
(no real edge, so the agent should at best learn to do nothing).

## Project layout

| File | What it does |
|---|---|
| `config.yaml` | exchange, symbol, timeframe, fees, split, training settings |
| `data/fetch.py` | downloads OHLCV via `ccxt` |
| `env/features.py` | stationary features: returns, volatility, RSI, volume ratio, MA distance, time of day |
| `env/trading_env.py` | Gymnasium env: action 0 = flat, 1 = long; reward = net log-return − costs |
| `common.py` | chronological split, train-only normalization, metrics |
| `train.py` | PPO training with validation checkpointing |
| `backtest.py` | metrics table + equity chart vs baselines |

## Design choices (and why)

- **Chronological split 70/15/15, never shuffled**: avoids training on the future.
- **Normalization uses train stats only**: no look-ahead leakage.
- **Fees (0.1%) + slippage (0.05%) per side, plus a trade penalty**: without this, agents over-trade.
- **Long/flat only**: spot, no shorting or leverage; suits a $10–20 account.
- **Stationary features, no raw prices**: prices in 2022 vs 2025 look nothing alike.

## Things to try next

1. Compare timeframes: set `timeframe: 1h` or `5m`, re-fetch, retrain.
2. More timesteps (1–5M) and several seeds — RL results vary a lot between seeds.
3. Walk-forward validation (train on months 1–12, test 13; slide forward).
4. Reward shaping: penalize drawdown / volatility (Sharpe-style reward).
5. Paper trading loop on the exchange testnet before any real money.
6. Live: hard caps in code (max $20 position, daily loss limit), outside the model.

## Notes

- If Binance is blocked in your country, set `exchange: binanceus`, `kraken` or `coinbase`
  (and `symbol: BTC/USD` for Kraken/Coinbase).
