# Trade_bot — RL crypto trading (learning project)

A PPO agent (Stable-Baselines3) that learns to go **long or flat** on BTC/USD
using **15-minute** candles, with Kraken's fees included in training.
Learns from BTC, ETH and SOL history (Coinbase), paper-trades BTC on live Kraken prices
(both available in Canada).

👉 **[COMMANDS.md](COMMANDS.md)**: every command and setting, plus experiments to try.

> Learning project. Expect break-even or small losses. Never risk money you can't lose.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

```bash
python -m data.fetch              # download ~4 years of BTC, ETH, SOL candles from Coinbase
python train.py                   # train 3 agents (seeds), keep the best one on the validation split
python backtest.py --split val    # compare vs buy & hold / do nothing
python paper_trade.py --replay    # watch it "trade" the held-out period bar by bar (fast demo)
python paper_trade.py             # paper trade on LIVE Kraken prices with $20 of fake money
streamlit run dashboard.py        # dashboard at http://localhost:8501 (run in a 2nd terminal)
```

`python backtest.py --split test` only once you're done tuning — the test set is your "unseen future".

### What the agent sees

Every candle it gets the last `window` candles of these features (all ratios/returns, never raw
prices, so they work on any coin and any year), each in hours so they work on any timeframe:

- **Momentum**: price change over 1 candle, 1h, 4h, 1 day, 4 days
- **Volatility**: 4h and 1-day volatility, and whether it's calmer or wilder than usual
- **Trend**: distance from the 1-day and 4-day average price
- **Range**: where price sits between its recent low and high (1 day, 4 days), distance from the 4-day high
- **Volume spikes**, **RSI**, **hour of day** and **day of week**
- **Its own position**: holding or not, profit/loss of the open trade, how long it's been open

### Learning from several coins

`train_symbols` (BTC, ETH, SOL by default) are all used for training; the bot still trades only
`symbol`. More variety forces it to learn patterns that work everywhere instead of memorizing
BTC's history. All coins are split on the **same dates** (no peeking at BTC's test period through
ETH), each coin is normalized with its own stats, and the best checkpoint is picked by the
**average** validation return across coins. The Training tab shows each coin separately.

### Trading rules and reward (config.yaml: `orders`, `rules`, `reward`)

Limit orders are on by default (Kraken maker fee); the rest are off. They apply identically to
training, backtests and paper trading.

- **Limit orders** (`orders.type: limit`): pay Kraken's 0.25% maker fee instead of 0.40% + slippage.
  The order sits at the candle's close and only fills if the next candle trades through it,
  so the bot sometimes misses a move, just like in real life.
- **Minimum holding time** (`rules.min_hold`): no selling for N candles after a buy, which kills
  quick in-and-out trades that only pay fees.
- **Stop-loss** (`rules.stop_loss`): sell at market if the open trade is down that much.
- **Excess reward** (`reward.type: excess`): the agent is scored on beating buy & hold instead of
  raw profit, so "just hold during a bull market" earns nothing and stepping aside in a crash pays.
- **Drawdown penalty** (`reward.drawdown_penalty`): extra penalty whenever equity sinks further
  below its peak.

### Walk-forward test (`python walkforward.py`)

One backtest period can make a bot look great or terrible by luck (our 15m validation was a -35%
crash, our 1h one a +80% rally). Walk-forward splits 40%..85% of history into several periods
and, for each one, trains fresh bots only on the data before it, picks checkpoints on the chunk
right before it, then trades it unseen. Result: in how many periods the bot beat buy & hold, the
average difference, and all periods chained together (dashboard tab **Walk-forward**).
The last 15% of history is never touched, so `backtest.py --split test` stays the final exam.

### Multiple seeds

RL results change a lot with the random seed: one run can look great purely by luck.
`train.py` trains `n_seeds` agents (3 by default, `--seeds 5` to change), prints a comparison,
and copies the best one (by validation return) to the main model used by `backtest.py` and
`paper_trade.py`. The **Training progress** tab shows one line per seed and a
"Seeds agree?" check: if seeds disagree a lot, don't trust the best one too much.
Training time = seeds × one run.

### Ensemble voting

After a multi-seed run, `backtest.py` and `paper_trade.py` use **all seeds together**. With
`ensemble.vote: soft` (default) the bot averages how confident each seed is that holding is right
and holds when the average is above 50%. `majority` is a strict yes/no vote (`min_agree`) that
proved too strict: seeds that buy at different times rarely agree, so it sat in cash.
Compare both with `python backtest.py --split val --vote majority`. Set `ensemble.enabled: false` to use only the best single seed.
The backtest table also shows each seed on its own so you can see the spread.

### Monkey test (skill or luck?)

`backtest.py` compares the agent with 1000 random traders that copy its **exact holding
periods** (same trades, same time in the market) but place them at random times, with the same
fees, so only the timing differs. If the agent doesn't beat ~95% of them, its timing
is probably no better than chance. Shown in the terminal and in the dashboard's Backtest tab.

### Dashboard tabs

- **🟢 Live (paper)**: candlestick chart with buy ▲ / sell ▼ markers, fake balance vs buy & hold,
  what the bot holds, trade log. Refreshes every 5 s. Switch between *Live prices* and *Replay (demo)*.
  - *Follow latest candles* ON: chart scrolls with the newest candles (pick how many to show).
  - OFF: chart pauses so you can zoom/pan freely (drag a box, scroll wheel, double-click to reset);
    numbers keep updating, click *Load new candles* to catch up.
- **Training progress**: live while `train.py` runs — one line per seed for validation return,
  trades, reward, drawdown; seed comparison when done.
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
| `env/features.py` | multi-horizon features: momentum (1h-4d), volatility regime, trend, range position, volume spikes, RSI, time |
| `env/trading_env.py` | Gymnasium env (one or several coins): action 0 = flat, 1 = long; sees position P&L and holding time |
| `common.py` | config, chronological split, train-only normalization, live observation, metrics |
| `train.py` | PPO training with validation checkpointing (+ saves normalization stats) |
| `backtest.py` | metrics vs baselines, each seed, and the monkey test |
| `agents.py` | loads the agent: ensemble of seeds (majority vote) or single model |
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
2. More timesteps (1–5M).
3. Walk-forward validation (train on months 1–12, test 13; slide forward).
4. Reward shaping: penalize drawdown / volatility (Sharpe-style reward).
5. Limit (maker) orders to pay 0.25% instead of 0.40%.
6. Live trading with real $20: API key in `.env` (never in git), withdrawals disabled on the key,
   hard caps in code (max position, daily loss limit).

## Notes

- Canada: Binance, Bybit and OKX left in 2023. Kraken, Coinbase and NDAX work.
- If Coinbase downloads fail, try `data_exchange: bitstamp`.
- Taxes: in Canada every crypto trade is a taxable event — keep `trades.csv`.
