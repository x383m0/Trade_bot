# Trade_bot — RL crypto trading (learning project)

A PPO agent (Stable-Baselines3) that learns to go **long or flat** on BTC/USD
using **15-minute** candles, with Kraken's fees included in training.
Trains on Coinbase history, paper-trades on live Kraken prices (both available in Canada).

> Learning project. Expect break-even or small losses. Never risk money you can't lose.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

```bash
python -m data.fetch              # download ~2 years of BTC/USD candles from Coinbase
python train.py                   # train PPO, keeps the best checkpoint on the validation split
python backtest.py --split val    # compare vs buy & hold / do nothing
python paper_trade.py --replay    # watch it "trade" the held-out period bar by bar (fast demo)
python paper_trade.py             # paper trade on LIVE Kraken prices with $20 of fake money
streamlit run dashboard.py        # dashboard at http://localhost:8501 (run in a 2nd terminal)
```

`python backtest.py --split test` only once you're done tuning — the test set is your "unseen future".

### Dashboard tabs

- **🟢 Live (paper)**: fake balance vs buy & hold, what the bot holds, buy/sell markers on price,
  trade log. Refreshes every 5 s. Switch between *Live prices* and *Replay (demo)*.
- **Training progress**: live while `train.py` runs — validation return, trades, reward, drawdown.
- **Backtest**: equity vs buy & hold, trades on price, position over time.

### Paper trading

- No API key, no real orders: it only reads public Kraken prices.
- Decides once per closed bar (every 15 min). Buys at the ask, sells at the bid, pays the Kraken fee.
- State is saved every bar: Ctrl+C and rerun to resume; `--reset` starts over with fresh fake cash.
- Your computer must stay on (and not sleep) for it to keep running.
- `--replay` options: `--delay 0.5` (slower), `--bars 500` (shorter).

Offline / pipeline test: `python -m data.fetch --synthetic` generates a random walk
(no real edge, so the agent should at best learn to do nothing).

## Project layout

| File | What it does |
|---|---|
| `config.yaml` | exchanges, symbol, timeframe, fees, split, training & paper settings |
| `data/fetch.py` | downloads OHLCV via `ccxt` from `data_exchange` |
| `env/features.py` | stationary features: returns, volatility, RSI, volume ratio, MA distance, time of day |
| `env/trading_env.py` | Gymnasium env: action 0 = flat, 1 = long; reward = net log-return − costs |
| `common.py` | config, chronological split, train-only normalization, live observation, metrics |
| `train.py` | PPO training with validation checkpointing (+ saves normalization stats) |
| `backtest.py` | metrics table + equity chart vs baselines |
| `paper_trade.py` | fake-money trading on live Kraken prices, or replay of the test period |
| `dashboard.py` | Streamlit dashboard |

## Comparing timeframes (15m vs 1h)

Set `timeframe: 1h` and `years: 8` in `config.yaml`, then fetch → train → backtest again.
Data, models and reports are stored per timeframe (`data/coinbase_BTCUSD_1h.parquet`,
`reports/1h/`...), so runs don't overwrite each other. The dashboard shows the timeframe
currently set in the config.

## Design choices (and why)

- **Chronological split 70/15/15, never shuffled**: avoids training on the future.
- **Normalization uses train stats only** (saved next to the model and reused live): no leakage.
- **Kraken fee 0.40% + slippage 0.05% per side, plus a trade penalty**: Kraken is ~4x pricier
  than Binance, so the agent must learn to trade rarely.
- **Long/flat only**: spot, no shorting or leverage; suits a $10–20 account.
- **Stationary features, no raw prices**: prices in 2022 vs 2025 look nothing alike.
- **Train on Coinbase, trade on Kraken**: Kraken's API only gives ~720 candles; BTC prices match
  closely across exchanges.

## Things to try next

1. Compare 15m vs 1h (see above).
2. More timesteps (1–5M) and several seeds — RL results vary a lot between seeds.
3. Walk-forward validation (train on months 1–12, test 13; slide forward).
4. Reward shaping: penalize drawdown / volatility (Sharpe-style reward).
5. Limit (maker) orders to pay 0.25% instead of 0.40%.
6. Live trading with real $20: API key in `.env` (never in git), withdrawals disabled on the key,
   hard caps in code (max position, daily loss limit).

## Notes

- Canada: Binance, Bybit and OKX left in 2023. Kraken, Coinbase and NDAX work.
- If Coinbase downloads fail, try `data_exchange: bitstamp`.
- Taxes: in Canada every crypto trade is a taxable event — keep `trades.csv`.
