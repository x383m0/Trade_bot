"""Interactive menu: every Trade_bot command in one place.

Usage:  python menu.py
Pick a number, it runs the command and shows you the exact command it ran (so you learn them).
Ctrl+C stops a running command and brings you back to the menu.
"""
import shlex
import subprocess
import sys
from pathlib import Path

from common import load_config, trade_rules

PY = sys.executable

MENU = [
    ("DATA", None),
    ("Download price history for all training coins", ["-m", "data.fetch"]),
    ("Download one coin only (asks which)", ["-m", "data.fetch", "--symbol", "{coin}"]),
    ("Check how much history exchanges have (asks which pair)", ["-m", "data.check_history", "{pair}"]),
    ("TRAINING", None),
    ("Quick test training (1 seed, 50k steps, ~minutes)", ["train.py", "--seeds", "1", "--timesteps", "50000"]),
    ("Medium training (3 seeds, 150k steps)", ["train.py", "--timesteps", "150000"]),
    ("Full training (settings from config.yaml)", ["train.py"]),
    ("Custom training (asks seeds and steps)", ["train.py", "--seeds", "{seeds}", "--timesteps", "{steps}"]),
    ("Train until I stop it (Ctrl+C), all seeds at once", ["train.py", "--timesteps", "0"]),
    ("FAST training on a strong PC: seeds at the same time (asks how many)",
     ["train.py", "--seeds", "{seeds}", "--parallel", "{seeds}"]),
    ("TESTING", None),
    ("Backtest on the validation period", ["backtest.py", "--split", "val"]),
    ("Backtest on another coin (asks which)", ["backtest.py", "--split", "val", "--symbol", "{coin}"]),
    ("FINAL EXAM: backtest on the test period (only when done tuning!)", ["backtest.py", "--split", "test"]),
    ("Walk-forward test: train + test on 6 periods (long! uses config.yaml)", ["walkforward.py"]),
    ("Walk-forward, quick check (3 periods, 100k steps)", ["walkforward.py", "--folds", "3",
                                                          "--timesteps", "100000"]),
    ("Simple rules vs buy & hold on the walk-forward periods (1-2 min, no training)", ["rules_test.py"]),
    ("Coin rotation (hold the strongest of BTC/ETH/SOL) vs buy & hold (1 min)", ["rotation_test.py"]),
    ("Download Fear & Greed + funding rate history (new information)", ["-m", "data.fetch_extra"]),
    ("Fear & Greed / funding rules vs buy & hold (needs the download above)", ["info_test.py"]),
    ("Grid bot vs buy & hold: are the 'steady profit' claims true? (10 s)", ["grid_test.py"]),
    ("PAPER TRADING (fake money)", None),
    ("Replay demo: watch the bot trade the test period (fresh $)", ["paper_trade.py", "--replay", "--reset"]),
    ("Replay demo, slow (0.5 s per candle)", ["paper_trade.py", "--replay", "--reset", "--delay", "0.5"]),
    ("Replay demo, short (300 candles)", ["paper_trade.py", "--replay", "--reset", "--bars", "300"]),
    ("LIVE paper trading on real Kraken prices (resumes; Ctrl+C to stop)", ["paper_trade.py"]),
    ("LIVE paper trading, start over with fresh fake money", ["paper_trade.py", "--reset"]),
    ("200-day rule: replay the final-exam period day by day (fresh $)", ["rule_trade.py", "--replay", "--reset"]),
    ("200-day rule: LIVE paper trading, one decision per day (resumes)", ["rule_trade.py"]),
    ("DASHBOARD", None),
    ("Open the dashboard (Ctrl+C here to stop it)", ["-m", "streamlit", "run", "dashboard.py"]),
    ("RESULTS", None),
    ("Show the last training summary (seed comparison)", "summary"),
    ("Show current settings", "settings"),
]

ASK = {"{coin}": ("Coin (e.g. ETH/USD)", "ETH/USD"), "{pair}": ("Pair (e.g. BTC/CAD)", "BTC/USD"),
       "{seeds}": ("Number of seeds", "3"), "{steps}": ("Steps per seed", "500000")}


def show_settings():
    cfg = load_config()
    tr = trade_rules(cfg)
    t = cfg["train"]
    print(f"""
  trades {cfg['symbol']} (live: {cfg['live_symbol']} on {cfg['exchange']}), learns from {', '.join(cfg['train_symbols'])}
  candles {cfg['timeframe']}, {cfg['years']} years of history
  orders {tr['order_type']} (fee {tr['maker_fee'] if tr['order_type'] == 'limit' else cfg['fee']}), \
min_hold {tr['min_hold']}, stop_loss {tr['stop_loss']}
  reward {tr['reward_type']}, drawdown_penalty {tr['drawdown_penalty']}, trade_penalty {cfg['trade_penalty']}
  training: {t.get('n_seeds', 1)} seeds x {t['total_timesteps']:,} steps, window {t['window']}, \
lr {t.get('learning_rate')}, gamma {t.get('gamma')}, ent_coef {t.get('ent_coef')}
  ensemble {'on' if (cfg.get('ensemble') or {}).get('enabled', True) else 'off'} \
(min_agree {(cfg.get('ensemble') or {}).get('min_agree', 0.5)})
  -> edit config.yaml to change these (see COMMANDS.md for what each does)""")


def show_summary():
    f = Path(load_config()["paths"]["report"]) / "seeds_summary.csv"
    print("\n" + (f.read_text() if f.exists() else f"  no training summary yet ({f})"))


def run(args):
    filled = []
    for a in args:
        if a in ASK:
            q, default = ASK[a]
            a = input(f"  {q} [{default}]: ").strip() or default
        filled.append(a)
    print(f"\n$ python {' '.join(shlex.quote(a) for a in filled)}\n")
    try:
        subprocess.run([PY, *filled])
    except KeyboardInterrupt:
        print("\n(stopped)")


def main():
    while True:
        cfg = load_config()
        print(f"\n===== Trade_bot menu  ({cfg['symbol']} {cfg['timeframe']}) =====")
        items = []
        for label, action in MENU:
            if action is None:
                print(f"\n {label}")
            else:
                items.append((label, action))
                print(f"  {len(items):>2}. {label}")
        print("\n   q. quit")
        choice = input("\nChoose a number: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if not choice.isdigit() or not 1 <= int(choice) <= len(items):
            print("  not a valid choice")
            continue
        label, action = items[int(choice) - 1]
        if action == "summary":
            show_summary()
        elif action == "settings":
            show_settings()
        else:
            run(action)
        input("\nPress Enter to go back to the menu...")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
