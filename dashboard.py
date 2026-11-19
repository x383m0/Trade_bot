"""Local dashboard: live paper trading, training progress and backtest results.

Usage: streamlit run dashboard.py
Leave it open while train.py / paper_trade.py run — those sections refresh automatically.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import load_config, read_csv, read_json, trade_rules

st.set_page_config(page_title="Trade_bot dashboard", page_icon="📈", layout="wide")
cfg = load_config()
REPORT = Path(cfg["paths"]["report"])
UP, DOWN = "#26a69a", "#ef5350"
# scroll wheel zooms; double-click resets the view
PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False}
SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
               "#17becf", "#bcbd22", "#7f7f7f"]

st.title("📈 Trade_bot dashboard")
st.caption(f"{cfg['symbol']} · {cfg['timeframe']} · trades on {cfg['exchange']} · "
           f"trained on {cfg['data_exchange']} data · "
           + (f"limit orders, fee {trade_rules(cfg)['maker_fee']*100:.2f}% per side"
              if trade_rules(cfg)["order_type"] == "limit" else
              f"fee {cfg['fee']*100:.2f}% + slippage {cfg['slippage']*100:.2f}% per side"))

tab_live, tab_train, tab_bt, tab_wf, tab_cfg = st.tabs(["🟢 Live (paper)", "Training progress",
                                                        "Backtest", "Walk-forward", "Config"])


def price_trace(d, price_col="close"):
    """Candlesticks if the data has OHLC columns, otherwise a plain line."""
    if {"open", "high", "low", "close"} <= set(d.columns):
        return go.Candlestick(x=d["time"], open=d["open"], high=d["high"], low=d["low"],
                              close=d["close"], name="BTC", showlegend=False,
                              increasing_line_color=UP, decreasing_line_color=DOWN)
    return go.Scatter(x=d["time"], y=d[price_col], name="price", line=dict(color="gray", width=1))


def trade_markers(fig, buys, sells, row):
    """Buys drawn just under the candle, sells just above (like most trading charts)."""
    for d, name, sym, color, col, k in ((buys, "buy", "triangle-up", UP, "low", 0.9985),
                                        (sells, "sell", "triangle-down", DOWN, "high", 1.0015)):
        if len(d):
            y = d[col] * k if col in d else d["marker_price"]
            fig.add_trace(go.Scatter(x=d["time"], y=y, mode="markers", name=name,
                                     marker=dict(symbol=sym, size=11, color=color,
                                                 line=dict(width=1, color="white")),
                                     text=d["hover"], hoverinfo="text"), row, 1)


# ---------------- Live paper trading (auto-refreshing) ----------------
with tab_live:
    @st.fragment(run_every="5s")   # re-checks every 5 s, so it notices paper_trade.py starting
    def live_section():
        labels = {"live": "RL bot · live prices", "replay": "RL bot · replay (demo)",
                  "rule_live": "200-day rule · live prices", "rule_replay": "200-day rule · replay (demo)"}
        modes = [m for m in labels if (REPORT / f"paper_{m}" / "equity.csv").exists()]
        if not modes:
            st.info("No paper trading yet. Run `python paper_trade.py` (RL bot, live Kraken prices), "
                    "`python paper_trade.py --replay` (fast demo on held-out data) or "
                    "`python rule_trade.py` (the 200-day rule).")
            return
        mode = st.radio("Account", modes, horizontal=True, key="paper_mode", format_func=labels.get)
        folder = REPORT / f"paper_{mode}"
        is_rule = mode.startswith("rule")
        who = "rule" if is_rule else "bot"
        eq = read_csv(folder / "equity.csv")
        state = read_json(folder / "state.json")
        if eq.empty or "equity" not in eq or not state:
            st.info("Started - waiting for the first closed bar…")
            return
        trades = read_csv(folder / "trades.csv")
        if "side" not in trades:              # trade log just being created
            trades = pd.DataFrame(columns=["time", "side", "price", "fee_usd"])
        last = eq.iloc[-1]
        start = state["start_cash"]
        pnl = last["equity"] - start

        c = st.columns(5)
        cur = state.get("currency", "USD")
        c[0].metric(f"Fake balance ({cur})", f"${last['equity']:.2f}", f"{pnl:+.2f} ({pnl / start * 100:+.1f}%)")
        c[1].metric("vs buy & hold", f"${last['buy_hold']:.2f}",
                    f"{last['equity'] - last['buy_hold']:+.2f}")
        c[2].metric("Holding", "₿ BTC" if state["pos"] else "💵 Cash",
                    f"agents: {state['votes']}" if state.get("votes") else None, delta_color="off",
                    help="With an ensemble, each trained seed votes every candle. The bot only "
                         "changes position when a majority agrees.")
        c[3].metric("Trades", state["n_trades"],
                    f"fees paid ${trades['fee_usd'].sum():.2f}" if len(trades) else None,
                    delta_color="off")
        c[4].metric("Last bar", str(last["time"])[5:], f"price {last['price']:,.2f}",
                    delta_color="off")
        if is_rule:
            st.caption("200-day rule: hold BTC when the daily close is more than 2% above its 200-day "
                       "average, cash when more than 2% below, otherwise keep the last position. One "
                       "decision per day at 00:00 UTC. Fake money, no orders sent."
                       + (" Replay = the final-exam period, day by day." if mode.endswith("replay") else
                          f" Live {cfg['exchange']} prices since {state['started']} UTC."))
        elif mode == "replay":
            st.caption("Replay = held-out test data fed one bar at a time, as if live. "
                       "Fake money, no orders sent.")
        else:
            st.caption(f"Live {cfg['exchange']} prices since {state['started']} UTC. "
                       "Fake money, no orders sent. One decision per closed bar.")

        # --- view controls ---
        ctl = st.columns([2, 3, 4])
        follow = ctl[0].toggle(
            "Follow latest candles", value=True, key="live_follow",
            help="ON: the chart scrolls by itself to show the newest candles.\n\n"
                 "OFF: the chart pauses so you can zoom and pan freely (drag a box, scroll wheel, "
                 "double-click to reset). The numbers above keep updating; click "
                 "'Load new candles' when you want the chart to catch up.")
        n_show = ctl[1].select_slider("Candles shown", [30, 50, 100, 200, 500, "All"], value=100,
                                      key="live_n", disabled=not follow)

        frozen_key = f"live_frozen_{mode}"
        if follow:
            st.session_state.pop(frozen_key, None)
            view = eq if n_show == "All" else eq.tail(int(n_show))
            ctl[2].caption("🖱️ Turn *Follow* off to pause the chart and zoom/pan freely")
        else:
            # freeze the chart at the candles we had when you paused: Plotly keeps your zoom
            # as long as the chart data doesn't change
            if ctl[2].button("⟳ Load new candles", key="live_load") or frozen_key not in st.session_state:
                st.session_state[frozen_key] = len(eq)
            view = eq.iloc[:st.session_state[frozen_key]]
            new = len(eq) - len(view)
            ctl[2].caption(f"⏸️ Chart paused · {new} new candle{'s' * (new != 1)} since · "
                           "drag / scroll to zoom · double-click to reset")

        # attach candle high/low to trades so markers sit under/over the candle
        tr = trades.merge(eq[[c for c in ("time", "high", "low") if c in eq]], on="time", how="left")
        tr["marker_price"] = tr["price"]
        # plain Python strings: works even when there are no trades (empty table)
        tr["hover"] = [f"{sd} @ {p:,.2f}<br>fee {fee:.3f}<br>{t}"
                       for sd, p, fee, t in zip(tr["side"], tr["price"], tr["fee_usd"], tr["time"])]
        tr = tr[tr["time"].isin(view["time"])].tail(1000)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                            vertical_spacing=0.05,
                            subplot_titles=(f"{state.get('symbol', cfg['symbol'])} {state.get('timeframe', cfg['timeframe'])} & trades",
                                            f"Fake balance ({cur})"))
        fig.add_trace(price_trace(view, "price"), 1, 1)
        rule = read_csv(folder / "rule.csv") if is_rule else pd.DataFrame()
        if "ma200" in rule:
            rv = view[["time"]].merge(rule, on="time", how="left")
            fig.add_trace(go.Scatter(x=rv["time"], y=rv["ma200"], name="200-day avg",
                                     line=dict(color="#7f7f7f", width=1.5)), 1, 1)
            for col, color, name in (("buy_line", UP, "buy line (+2%)"), ("sell_line", DOWN, "sell line (-2%)")):
                fig.add_trace(go.Scatter(x=rv["time"], y=rv[col], name=name,
                                         line=dict(color=color, width=1, dash="dot")), 1, 1)
        trade_markers(fig, tr[tr["side"] == "BUY"], tr[tr["side"] == "SELL"], 1)
        fig.add_trace(go.Scatter(x=view["time"], y=view["equity"], name=who,
                                 line=dict(color="#1f77b4", width=2)), 2, 1)
        fig.add_trace(go.Scatter(x=view["time"], y=view["buy_hold"], name="buy & hold",
                                 line=dict(color="#ff7f0e", width=1.5, dash="dot")), 2, 1)
        fig.add_hline(y=start, line_dash="dash", line_color="gray", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        # uirevision: while following, the view resets only when a NEW candle arrives;
        # in free mode it never resets, so your zoom survives every refresh.
        rev = f"follow-{n_show}-{last['time']}" if follow else f"free-{mode}"
        fig.update_layout(height=680, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.05), uirevision=rev,
                          hovermode="x unified" if follow else "closest")
        st.plotly_chart(fig, width="stretch", key="live_chart", config=PLOT_CONFIG)

        st.subheader("Trade log")
        if len(trades):
            st.dataframe(trades.iloc[::-1], width="stretch", hide_index=True)
        else:
            st.caption(f"No trades yet - the {who} is staying in cash.")

    live_section()


# ---------------- Training progress (auto-refreshing) ----------------
def seed_chart(df, col, title, yaxis, zero_line=False, best=None):
    """One line per seed; the chosen/best seed drawn thicker."""
    fig = go.Figure()
    for i, (seed, d) in enumerate(df.groupby("seed")):
        thick = best is not None and seed == best
        color = SEED_COLORS[i % len(SEED_COLORS)]
        fig.add_trace(go.Scatter(x=d["timesteps"], y=d[col], name=f"seed {seed}" + (" ★" if thick else ""),
                                 mode="lines+markers", opacity=1 if thick else 0.7,
                                 line=dict(width=3.5 if thick else 1.5, color=color),
                                 marker=dict(size=7 if thick else 4, color=color)))
    if zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, yaxis_title=yaxis, height=320,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom"),
                      uirevision=title)  # keeps your zoom when the chart refreshes
    return fig


with tab_train:
    @st.fragment(run_every="5s")
    def training_section():
        logs = sorted(REPORT.glob("training_log_seed*.csv"), key=lambda f: f.stat().st_mtime)
        frames = [d for d in (read_csv(f) for f in logs) if not d.empty and "seed" in d]
        if not frames:
            st.info("No training yet. Run `python train.py` — this page updates live."
                    if not logs else "Training started, waiting for the first validation checkpoint…")
            return
        df = pd.concat(frames, ignore_index=True)
        run_f, sum_f = REPORT / "train_run.json", REPORT / "seeds_summary.csv"
        run = read_json(run_f) or {"seeds": sorted(df["seed"].unique()),
                                   "timesteps": cfg["train"]["total_timesteps"]}
        seeds, target = run["seeds"], run["timesteps"]
        finished = sum_f.exists()

        cur_seed = int(frames[-1]["seed"].iloc[0])           # most recently written log
        cur = df[df["seed"] == cur_seed]
        k = seeds.index(cur_seed) + 1 if cur_seed in seeds else len(df["seed"].unique())
        if not target:   # python train.py --timesteps 0: runs until Ctrl+C
            most = int(df["timesteps"].max())
            st.progress(1.0 if finished else min(0.99, most / 5_000_000),
                        text="✅ Training finished" if finished else
                        f"♾️ Unlimited training · {len(df['seed'].unique())} seeds · furthest at "
                        f"{most:,} steps · {cur['elapsed_s'].iloc[-1] / 60:.0f} min · "
                        "press Ctrl+C in the training terminal to stop")
        else:
            steps_done = sum(min(int(df.loc[df["seed"] == s, "timesteps"].max()), target)
                             for s in df["seed"].unique())
            overall = min(1.0, steps_done / (target * len(seeds)))
            st.progress(overall, text="✅ Training finished" if finished else
                        f"Seed {k}/{len(seeds)} (seed {cur_seed}) · "
                        f"{int(cur['timesteps'].iloc[-1]):,} / {target:,} steps · "
                        f"overall {overall * 100:.0f}% · {cur['elapsed_s'].iloc[-1] / 60:.1f} min on this seed")

        best_per_seed = df.loc[df.groupby("seed")["val_return_%"].idxmax()]
        top = best_per_seed.loc[best_per_seed["val_return_%"].idxmax()]
        last = cur.iloc[-1]
        c = st.columns(4)
        c[0].metric(f"Latest val return (seed {cur_seed})", f"{last['val_return_%']:+.2f}%",
                    f"{last['val_return_%'] - cur['val_return_%'].iloc[-2]:+.2f}" if len(cur) > 1 else None)
        c[1].metric("Best so far (all seeds)", f"{top['val_return_%']:+.2f}%",
                    f"seed {int(top['seed'])} at {int(top['timesteps']):,} steps", delta_color="off")
        c[2].metric(f"Val trades (seed {cur_seed})", int(last["val_trades"]),
                    f"{int(last['val_trades'] - cur['val_trades'].iloc[-2]):+d}" if len(cur) > 1 else None,
                    delta_color="inverse")
        if len(best_per_seed) > 1:
            r = best_per_seed["val_return_%"]
            agree = "✅ consistent" if r.std() <= max(2.0, abs(r.mean())) else "⚠️ mostly luck"
            c[3].metric("Seeds agree?", agree, f"spread ±{r.std():.1f}% around {r.mean():+.1f}%",
                        delta_color="off",
                        help="If different random seeds give very different results, the 'best' one "
                             "probably just got lucky on the validation period.")
        else:
            c[3].metric("Val Sharpe", f"{last['val_sharpe']:.2f}")

        best_seed = int(top["seed"])
        charts = [("val_return_%", "Validation return", "%", True),
                  ("val_trades", "Trades on validation (fewer = less fee drag)", "trades", False),
                  ("train_ep_reward", "Training reward", "reward", True),
                  ("val_max_drawdown_%", "Validation max drawdown", "%", False)]
        cols = st.columns(2) + st.columns(2)
        for i, (col, (y, title, unit, zero)) in enumerate(zip(cols, charts)):
            col.plotly_chart(seed_chart(df, y, title, unit, zero, best_seed), width="stretch",
                             key=f"train_chart_{i}", config=PLOT_CONFIG)
        st.caption("x-axis = training steps. Each line is one agent trained with a different "
                   "random seed. Thick line ★ = best seed so far."
                   + (" Validation numbers are averaged over all coins." if
                      any(c.startswith("val_ret_") for c in df.columns) else ""))

        coin_cols = [c for c in df.columns if c.startswith("val_ret_")]
        if len(coin_cols) > 1:
            d = df[df["seed"] == best_seed]
            fig = go.Figure()
            for i, col in enumerate(coin_cols):
                fig.add_trace(go.Scatter(x=d["timesteps"], y=d[col], name=col[8:], mode="lines+markers",
                                         line=dict(color=SEED_COLORS[(i + 5) % len(SEED_COLORS)])))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(title=f"Validation return per coin (seed {best_seed} ★)", yaxis_title="%",
                              height=300, margin=dict(l=10, r=10, t=40, b=10), uirevision="coins",
                              legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom"))
            st.plotly_chart(fig, width="stretch", key="coin_chart", config=PLOT_CONFIG)
            st.caption("If it only works on one coin, it probably memorized that coin. "
                       "A real pattern should help on all of them.")

        summ = read_csv(sum_f) if finished else pd.DataFrame()
        if "chosen" in summ and summ["chosen"].astype(bool).any():
            summ["chosen"] = summ["chosen"].astype(bool)
            st.subheader("Seed comparison")
            chosen = int(summ.loc[summ["chosen"], "seed"].iloc[0])
            st.markdown(f"Best checkpoint of each seed on validation. **Seed {chosen}** was "
                        "chosen and copied to the main model used by `backtest.py` and "
                        "`paper_trade.py`.")
            bar = go.Figure(go.Bar(x=[f"seed {s}" for s in summ["seed"]], y=summ["val_return_%"],
                                   marker_color=["#1f77b4" if ch else "#aec7e8" for ch in summ["chosen"]],
                                   text=summ["val_return_%"].map("{:+.2f}%".format), textposition="outside"))
            bar.add_hline(y=summ["val_return_%"].mean(), line_dash="dot", line_color="gray",
                          annotation_text="mean")
            bar.update_layout(height=300, yaxis_title="best val return %",
                              margin=dict(l=10, r=10, t=30, b=10), uirevision="seeds")
            l, r = st.columns([2, 3])
            l.plotly_chart(bar, width="stretch", key="seed_bar", config=PLOT_CONFIG)
            r.dataframe(summ, width="stretch", hide_index=True)

        with st.expander("Raw log"):
            st.dataframe(df, width="stretch", hide_index=True)

    training_section()


# ---------------- Backtest ----------------
def backtest_section():
    splits = [s for s in ("val", "test") if (REPORT / f"equity_{s}.csv").exists()]
    if not splits:
        st.info("No backtest yet. Run `python backtest.py --split val`.")
    else:
        split = st.radio("Split", splits, horizontal=True)
        eq = read_csv(REPORT / f"equity_{split}.csv", parse_dates=["time"])
        mt = read_csv(REPORT / f"metrics_{split}.csv", index_col=0)
        if eq.empty or "PPO agent" not in mt.index or "Buy & hold" not in mt.index:
            st.info("This backtest's files are incomplete (still being written, or the run was "
                    f"stopped). Run `python backtest.py --split {split}` again.")
            return

        a, b = mt.loc["PPO agent"], mt.loc["Buy & hold"]
        c = st.columns(4)
        c[0].metric("Agent return", f"{a['total_return_%']:+.2f}%",
                    f"{a['total_return_%'] - b['total_return_%']:+.2f} vs buy & hold")
        c[1].metric("Agent Sharpe", f"{a['sharpe']:.2f}", f"{a['sharpe'] - b['sharpe']:+.2f} vs B&H")
        c[2].metric("Max drawdown", f"{a['max_drawdown_%']:.2f}%")
        c[3].metric("Trades", int(a["trades"]))
        if split == "test":
            st.warning("Test split = your unseen future. Look at it once you're done tuning, not during.")
        st.caption("🖱️ Drag to zoom · scroll wheel to zoom · double-click to reset. "
                   "Zoom in to see individual candles and trades.")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.35, 0.15],
                            vertical_spacing=0.04,
                            subplot_titles=("Price & trades", "Equity (start = 1.0)", "Position"))
        fig.add_trace(price_trace(eq), 1, 1)
        chg = eq["position"].diff().fillna(eq["position"])
        buys, sells = eq[chg > 0].copy(), eq[chg < 0].copy()
        for d, side in ((buys, "BUY"), (sells, "SELL")):
            d["marker_price"] = d["close"]
            d["hover"] = [f"{side} @ {c:,.2f}<br>{t}" for c, t in zip(d["close"], d["time"])]
        if len(buys) + len(sells) <= 3000:   # too many markers makes the chart unreadable
            trade_markers(fig, buys, sells, 1)
        fig.add_trace(go.Scatter(x=eq["time"], y=eq["agent"], name="PPO agent"), 2, 1)
        fig.add_trace(go.Scatter(x=eq["time"], y=eq["buy_hold"], name="Buy & hold", opacity=0.7), 2, 1)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_trace(go.Scatter(x=eq["time"], y=eq["position"], name="position", fill="tozeroy",
                                 line=dict(shape="hv", width=0.5), showlegend=False), 3, 1)
        fig.update_yaxes(tickvals=[0, 1], ticktext=["flat", "long"], row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(height=850, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h"), uirevision=split)
        st.plotly_chart(fig, width="stretch", key="bt_chart", config=PLOT_CONFIG)

        st.subheader("Metrics")
        mk_f = REPORT / f"monkey_{split}.json"
        mk = read_json(mk_f)
        label = mk["agent_label"] if mk else "PPO agent"
        if mk and mk.get("symbol"):
            st.caption(f"Coin tested: **{mk['symbol']}** (`python backtest.py --symbol ETH/USD` to test another)")
        st.dataframe(mt.rename(index={"PPO agent": f"▶ {label}"}), width="stretch")
        st.caption(f"Time in market: {eq['position'].mean()*100:.1f}% of bars. "
                   "Rows 'seed …' = each trained agent on its own; a big spread between them "
                   "means results depend on luck.")

        if mk:
            st.subheader("🐒 Monkey test: skill or luck?")
            if mk["trades"] == 0:
                st.info("The agent made no trades on this period, so there's nothing to compare.")
            elif not mk.get("testable", True):
                st.info(f"Can't judge: only {mk.get('holding_periods', '?')} holding period(s), so "
                        "there's no timing to shuffle - every random trader ends up identical to "
                        "the agent (it's basically buy & hold).")
            else:
                if mk.get("holding_periods", 99) < 5:
                    st.caption(f"⚠️ Only {mk['holding_periods']} holding periods - weak test.")
                beats = mk["beats_%"]
                verdict = ("✅ likely real skill" if beats >= 95 else
                           "🤔 maybe something, not convincing" if beats >= 75 else
                           "❌ no better than chance")
                c = st.columns(3)
                c[0].metric("Beats random traders", f"{beats:.1f}%", verdict, delta_color="off")
                c[1].metric("Agent return", f"{mk['agent_return_%']:+.2f}%")
                c[2].metric("Typical monkey", f"{mk['monkey_median_%']:+.2f}%",
                            f"90% between {mk['monkey_p5_%']:+.2f}% and {mk['monkey_p95_%']:+.2f}%",
                            delta_color="off")
                st.caption(f"{mk['runs']} random traders each copied the agent's holding periods "
                           f"exactly ({mk['trades']} trades, same time in the market) but placed them "
                           "at random times, paying the same fees. So only the TIMING differs. If "
                           "the agent's timing were no better than chance, it would land in the "
                           "middle of the pack. Aim for 95%+.")
                mf = REPORT / f"monkeys_{split}.csv"
                if mf.exists():
                    mon = read_csv(mf).get("return_%", pd.Series(dtype=float))
                    h = go.Figure(go.Histogram(x=mon, nbinsx=50, name="random traders",
                                               marker_color="#c7c7c7"))
                    h.add_vline(x=mk["agent_return_%"], line_color="#1f77b4", line_width=3,
                                annotation_text=f"agent {mk['agent_return_%']:+.2f}%",
                                annotation_position="top")
                    h.add_vline(x=float(mon.quantile(0.95)), line_dash="dot", line_color="gray",
                                annotation_text="95% line", annotation_position="bottom right")
                    h.update_layout(height=300, xaxis_title="return % over the period",
                                    yaxis_title="# random traders", showlegend=False,
                                    margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(h, width="stretch", key="monkey_hist", config=PLOT_CONFIG)


with tab_bt:
    backtest_section()


# ---------------- Walk-forward ----------------
with tab_wf:
    @st.fragment(run_every="10s")
    def walkforward_section():
        wf_dir = REPORT / "walkforward"
        fold_files = sorted(wf_dir.glob("fold*/result.json"), key=lambda f: int(f.parent.name[4:]))
        if not fold_files:
            st.info("No walk-forward test yet. Run `python walkforward.py` - it trains and tests the "
                    "bot on several periods in a row, so one lucky backtest can't fool you.")
            return
        run = read_json(wf_dir / "run.json", default={})
        res = pd.DataFrame([r for r in (read_json(f) for f in fold_files) if r])
        if run.get("signature") and "signature" in res:
            res = res[res["signature"] == run["signature"]]   # only folds of the current settings
        if res.empty:
            st.info("Walk-forward started with new settings - waiting for the first period to finish.")
            return
        total = run.get("folds", len(res))
        won = int((res["bot_%"] > res["buy_hold_%"]).sum())
        bot_total = (np.prod(1 + res["bot_%"] / 100) - 1) * 100
        bh_total = (np.prod(1 + res["buy_hold_%"] / 100) - 1) * 100
        if len(res) < total:
            st.progress(len(res) / total, text=f"{len(res)} of {total} periods done (training continues "
                                               "in the terminal; finished periods are kept)")
        c = st.columns(4)
        c[0].metric("Beat buy & hold", f"{won} / {len(res)} periods",
                    "✅ consistent" if won / len(res) >= 0.7 else "🤔 mixed" if won / len(res) >= 0.4
                    else "❌ mostly loses", delta_color="off")
        c[1].metric("Avg difference per period", f"{res['excess_%'].mean():+.1f}%",
                    help="Bot return minus buy & hold return, averaged over periods")
        c[2].metric("All periods chained", f"{bot_total:+.1f}%", f"buy & hold {bh_total:+.1f}%",
                    delta_color="off")
        c[3].metric("Worst drop in a period", f"{res['bot_max_dd_%'].min():.1f}%",
                    f"buy & hold {res['bh_max_dd_%'].min():.1f}%", delta_color="off")
        st.caption("Each period: fresh bots trained only on data BEFORE it, then trading it unseen. "
                   "The last part of history (the final exam) is never used here.")

        labels = [f"{r.test_start}→{r.test_end}" for r in res.itertuples()]
        bar = go.Figure()
        bar.add_trace(go.Bar(x=labels, y=res["bot_%"], name="bot", marker_color="#1f77b4",
                             text=res["bot_%"].map("{:+.1f}%".format), textposition="outside"))
        bar.add_trace(go.Bar(x=labels, y=res["buy_hold_%"], name="buy & hold", marker_color="#ff7f0e",
                             text=res["buy_hold_%"].map("{:+.1f}%".format), textposition="outside"))
        if "rule_%" in res and res["rule_%"].notna().all():
            bar.add_trace(go.Bar(x=labels, y=res["rule_%"], name="200-day rule", marker_color="#7f7f7f",
                                 text=res["rule_%"].map("{:+.1f}%".format), textposition="outside"))
        bar.update_layout(barmode="group", height=360, yaxis_title="return % in the period",
                          margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
                          uirevision="wf_bar")
        st.plotly_chart(bar, width="stretch", key="wf_bar", config=PLOT_CONFIG)

        # stitched equity: each period starts where the previous one ended
        curves, b_lvl, h_lvl, r_lvl = [], 1.0, 1.0, 1.0
        for f in fold_files:
            ef = f.parent / "equity.csv"
            if not ef.exists() or f.parent.name[4:] not in map(str, res["fold"]):
                continue
            e = read_csv(ef)
            if e.empty or "bot" not in e:
                continue
            e["bot"], e["buy_hold"] = e["bot"] * b_lvl, e["buy_hold"] / e["buy_hold"].iloc[0] * h_lvl
            b_lvl, h_lvl = e["bot"].iloc[-1], e["buy_hold"].iloc[-1]
            if "rule_200d" in e:
                e["rule_200d"] = e["rule_200d"] * r_lvl
                r_lvl = e["rule_200d"].iloc[-1]
            curves.append(e)
        if curves:
            eq = pd.concat(curves)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq["time"], y=eq["bot"], name="bot", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=eq["time"], y=eq["buy_hold"], name="buy & hold",
                                     line=dict(color="#ff7f0e", dash="dot")))
            if "rule_200d" in eq and eq["rule_200d"].notna().all():
                fig.add_trace(go.Scatter(x=eq["time"], y=eq["rule_200d"], name="200-day rule",
                                         line=dict(color="#7f7f7f", width=1.5)))
            fig.add_hline(y=1, line_dash="dash", line_color="gray")
            fig.update_layout(title="All test periods chained (start = 1.0)", height=360,
                              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"),
                              uirevision="wf_eq")
            st.plotly_chart(fig, width="stretch", key="wf_eq", config=PLOT_CONFIG)

        st.subheader("Periods")
        cols = ["fold", "test_start", "test_end", "bot_%", "buy_hold_%", "rule_%", "excess_%", "trades",
                "in_market_%", "bot_max_dd_%", "bh_max_dd_%", "rule_max_dd_%", "monkey_beats_%"]
        st.dataframe(res[[c for c in cols if c in res]], width="stretch", hide_index=True)

    walkforward_section()


# ---------------- Config ----------------
with tab_cfg:
    st.json(cfg)
