"""Local dashboard: live paper trading, training progress and backtest results.

Usage: streamlit run dashboard.py
Leave it open while train.py / paper_trade.py run — those sections refresh automatically.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import load_config

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
           f"fee {cfg['fee']*100:.2f}% + slippage {cfg['slippage']*100:.2f}% per side")

tab_live, tab_train, tab_bt, tab_cfg = st.tabs(["🟢 Live (paper)", "Training progress",
                                                "Backtest", "Config"])


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
        modes = [m for m in ("live", "replay") if (REPORT / f"paper_{m}" / "equity.csv").exists()]
        if not modes:
            st.info("No paper trading yet. Run `python paper_trade.py` (live Kraken prices) or "
                    "`python paper_trade.py --replay` (fast demo on held-out data).")
            return
        mode = st.radio("Mode", modes, horizontal=True, key="paper_mode",
                        format_func=lambda m: {"live": "Live prices", "replay": "Replay (demo)"}[m])
        folder = REPORT / f"paper_{mode}"
        eq = pd.read_csv(folder / "equity.csv")
        if eq.empty or not (folder / "state.json").exists():
            st.info("Started - waiting for the first closed bar…")
            return
        state = json.loads((folder / "state.json").read_text())
        trades = pd.read_csv(folder / "trades.csv")
        last = eq.iloc[-1]
        start = state["start_cash"]
        pnl = last["equity"] - start

        c = st.columns(5)
        c[0].metric("Fake balance", f"${last['equity']:.2f}", f"{pnl:+.2f} ({pnl / start * 100:+.1f}%)")
        c[1].metric("vs buy & hold", f"${last['buy_hold']:.2f}",
                    f"{last['equity'] - last['buy_hold']:+.2f}")
        c[2].metric("Holding", "₿ BTC" if state["pos"] else "💵 Cash")
        c[3].metric("Trades", state["n_trades"],
                    f"fees paid ${trades['fee_usd'].sum():.2f}" if len(trades) else None,
                    delta_color="off")
        c[4].metric("Last bar", str(last["time"])[5:], f"price {last['price']:,.2f}",
                    delta_color="off")
        if mode == "replay":
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
        tr["hover"] = (tr["side"] + " @ " + tr["price"].map("{:,.2f}".format) + "<br>fee $"
                       + tr["fee_usd"].map("{:.3f}".format) + "<br>" + tr["time"])
        tr = tr[tr["time"].isin(view["time"])].tail(1000)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                            vertical_spacing=0.05,
                            subplot_titles=(f"{cfg['symbol']} {cfg['timeframe']} & trades",
                                            "Fake balance ($)"))
        fig.add_trace(price_trace(view, "price"), 1, 1)
        trade_markers(fig, tr[tr["side"] == "BUY"], tr[tr["side"] == "SELL"], 1)
        fig.add_trace(go.Scatter(x=view["time"], y=view["equity"], name="bot",
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
            st.caption("No trades yet - the bot is staying in cash.")

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
        frames = [d for d in (pd.read_csv(f) for f in logs) if not d.empty]
        if not frames:
            st.info("No training yet. Run `python train.py` — this page updates live."
                    if not logs else "Training started, waiting for the first validation checkpoint…")
            return
        df = pd.concat(frames, ignore_index=True)
        run_f, sum_f = REPORT / "train_run.json", REPORT / "seeds_summary.csv"
        run = (json.loads(run_f.read_text()) if run_f.exists()
               else {"seeds": sorted(df["seed"].unique()), "timesteps": cfg["train"]["total_timesteps"]})
        seeds, target = run["seeds"], run["timesteps"]
        finished = sum_f.exists()

        cur_seed = int(frames[-1]["seed"].iloc[0])           # most recently written log
        cur = df[df["seed"] == cur_seed]
        steps_done = sum(min(int(df.loc[df["seed"] == s, "timesteps"].max()), target)
                         for s in df["seed"].unique())
        overall = min(1.0, steps_done / (target * len(seeds)))
        k = seeds.index(cur_seed) + 1 if cur_seed in seeds else len(df["seed"].unique())
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
                   "random seed. Thick line ★ = best seed so far.")

        if finished:
            summ = pd.read_csv(sum_f)
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
with tab_bt:
    splits = [s for s in ("val", "test") if (REPORT / f"equity_{s}.csv").exists()]
    if not splits:
        st.info("No backtest yet. Run `python backtest.py --split val`.")
    else:
        split = st.radio("Split", splits, horizontal=True)
        eq = pd.read_csv(REPORT / f"equity_{split}.csv", parse_dates=["time"])
        mt = pd.read_csv(REPORT / f"metrics_{split}.csv", index_col=0)

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
            d["hover"] = side + " @ " + d["close"].map("{:,.2f}".format) + "<br>" + d["time"].astype(str)
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
        st.dataframe(mt, width="stretch")
        st.caption(f"Time in market: {eq['position'].mean()*100:.1f}% of bars")


# ---------------- Config ----------------
with tab_cfg:
    st.json(cfg)
