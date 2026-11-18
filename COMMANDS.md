# Trade_bot — command cheat sheet

> **Easiest:** `python menu.py` shows every command below as a numbered menu, runs the one you
> pick, and prints the exact command it ran. Ctrl+C stops a command and returns to the menu.
>
> **Linux / Mac:** activate with `source .venv/bin/activate`, print files with `cat` instead of
> `type`, and save output with `| tee file.txt` instead of `| Tee-Object -FilePath file.txt`.

All commands run in **PowerShell**, inside the `Trade_bot` folder, with the environment active.

## Every time you open a new terminal

```powershell
cd Trade_bot
.venv\Scripts\Activate.ps1
```

Get the latest code: `git pull` (then `pip install -r requirements.txt` if it says new packages).

---

## 1. Data

| Command | What it does |
|---|---|
| `python -m data.fetch` | Download history for every coin in `train_symbols` (a few min per coin) |
| `python -m data.fetch --symbol ETH/USD` | Download just one coin |
| `python -m data.fetch --synthetic` | Fake random data (offline testing; the bot can't learn anything real from it) |

Rerun after changing `symbol`, `train_symbols`, `timeframe` or `years`.

## 2. Training

| Command | What it does |
|---|---|
| `python train.py` | Train `n_seeds` agents (3) for `total_timesteps` (500k) each |
| `python train.py --timesteps 100000` | Quick run (good for trying settings) |
| `python train.py --seeds 5` | Train 5 agents instead of 3 (more reliable, longer) |
| `python train.py --seeds 1 --timesteps 50000` | Fastest possible run to check things work |

Watch it live in the dashboard's **Training progress** tab. Stop anytime with `Ctrl+C`
(each seed keeps its best checkpoint, but the ensemble/summary is only written at the end).

## 3. Testing the model

| Command | What it does |
|---|---|
| `python backtest.py --split val` | Test on the validation period (use this while experimenting) |
| `python backtest.py --split val --symbol ETH/USD` | Test on another coin |
| `python backtest.py --split test` | **Final exam** — only once you're done tuning |
| `python paper_trade.py --replay` | Watch the bot trade the test period candle by candle |
| `python paper_trade.py --replay --reset` | Same, starting over with fresh $20 |
| `python paper_trade.py --replay --delay 0.5` | Slower replay (0.5 s per candle) |
| `python paper_trade.py --replay --bars 300` | Shorter replay (300 candles) |
| `python paper_trade.py` | **Live** paper trading on real Kraken prices (fake money) |
| `python paper_trade.py --reset` | Live, starting over with fresh fake $20 |

What to look at in a backtest:
- **PPO agent vs Buy & hold**: does it beat just holding?
- **seed rows**: do the seeds agree? Big spread = luck.
- **Monkey test**: beats 95%+ of random traders = likely real skill.

## 4. Dashboard

```powershell
streamlit run dashboard.py
```
Opens http://localhost:8501. Run it in a **second** PowerShell window. `Ctrl+C` to stop.

---

## 5. Settings you can change (`config.yaml`)

Open it in any text editor (Notepad, VS Code). After saving, rerun what the table says.

| Setting | Default | What it does | Then rerun |
|---|---|---|---|
| `symbol` | BTC/USD | Coin the bot trades (its training data) | fetch, train |
| `live_symbol` | BTC/CAD | Pair used for live paper trading on Kraken (BTC/CAD = Canadian dollars) | paper `--reset` |
| `train_symbols` | BTC, ETH, SOL | Coins it learns from. Add e.g. `- XRP/USD`, `- LTC/USD` | fetch, train |
| `timeframe` | 15m | Candle size: `5m`, `15m`, `1h` | fetch, train |
| `years` | 4 | History to download (use ~8 with `1h`) | fetch, train |
| `fee` | 0.004 | Fee per trade (0.0025 = Kraken limit-order fee) | train |
| `trade_penalty` | 0.0002 | Higher = bot trades less | train |
| `orders.type` | limit | `limit` = 0.25% maker fee, but orders can miss; `market` = 0.40% + slippage, always fills | train |
| `rules.min_hold` | 0 | Candles it must hold before selling (24 on 1h = 1 day, 96 on 15m = 1 day) | train |
| `rules.stop_loss` | 0.0 | Sell at market if the trade is down this much (0.05 = 5%) | train |
| `reward.type` | profit | `excess` = scored on beating buy & hold, not raw profit | train |
| `reward.drawdown_penalty` | 0.0 | Punish sinking below its peak (try 0.5–2) | train |
| `train.total_timesteps` | 500000 | Training length per seed | train |
| `train.n_seeds` | 3 | Agents per run | train |
| `train.window` | 16 | Candles the agent looks back at | train |
| `train.learning_rate` | 0.0001 | Lower = slower but steadier learning (0.0003 was unstable) | train |
| `train.gamma` | 0.99 | How far ahead it plans (0.995 = longer-term) | train |
| `train.ent_coef` | 0.01 | Exploration (higher = tries more things) | train |
| `ensemble.enabled` | true | false = use only the best seed | backtest / paper |
| `ensemble.min_agree` | 0.5 | Higher (e.g. 0.7) = needs more agents to agree → fewer trades | backtest / paper |
| `paper.starting_cash` | 20 | Fake money for paper trading | paper `--reset` |

## 6. Experiments to try

Change one thing at a time, then compare `python backtest.py --split val` results.
Save each result with its own name so you can compare later, e.g.
`python backtest.py --split val | Tee-Object -FilePath run2_costs.txt`

### Recommended plan (after the 15m verdict: fees + holding through crashes)

Limit orders on Kraken are the default (`orders.type: limit`, 0.25% maker fee).

The bot makes every decision itself: no forced holding time (`min_hold: 0`) and no stop-loss
(`stop_loss: 0`). Instead, training nudges it to learn caution on its own.

| Run | Change in config.yaml | Then |
|---|---|---|
| 1. 1h + limit orders | `timeframe: 1h`, `years: 8` | fetch, train, backtest |
| 2. + learn caution, steadier learning | `trade_penalty: 0.002` (10x; training-only cost), `train.learning_rate: 0.0001` | train, backtest |
| 3. + smarter reward | `reward.type: excess`, `reward.drawdown_penalty: 1.0` | train, backtest |

Why caution matters: with Kraken's fees, trades shorter than ~1 day rarely win even with very good
predictions (a 1h move is ~0.5%, a limit round trip costs ~0.5%). The hope is that the bot learns
to trade less and hold longer by itself.

Keep the changes from the previous run each time. Each run's question: did it beat the previous
one **against buy & hold**, with fewer trades and a smaller drawdown?

**1h candles instead of 15m** (often better with Kraken's fees):
```yaml
timeframe: 1h
years: 8
```
→ `python -m data.fetch`, `python train.py`, `python backtest.py --split val`.
15m and 1h results are stored separately; the dashboard shows the one set in the config.

**Make it trade less:** `trade_penalty: 0.001` or `ensemble.min_agree: 0.7` → retrain / re-backtest.

**What if fees were lower (limit orders)?** `fee: 0.0025` → retrain.

**Learn from more coins:** add `- XRP/USD` / `- LTC/USD` / `- DOGE/USD` under `train_symbols` → fetch, train.

**Only BTC (to compare with multi-coin):** keep just `- BTC/USD` under `train_symbols` → train.

**Train longer:** `total_timesteps: 1500000` (about 3× longer).

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named ...` | Activate the venv (`.venv\Scripts\Activate.ps1`), then `pip install -r requirements.txt` |
| `... not found - run python -m data.fetch` | Download data first |
| `run python train.py first` / `older version` | Train again (new features need a new model) |
| `Log files are from an older version` | Add `--reset` to the paper_trade command |
| A coin fails to download | Check it exists on Coinbase as `XXX/USD`; the others still download |
| Dashboard shows old numbers | It shows the `timeframe` currently in `config.yaml` |
