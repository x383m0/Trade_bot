"""Local dashboard for training progress and backtest results.

Usage: streamlit run dashboard.py
Leave it open while `python train.py` runs — the training section refreshes automatically.
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

st.title("📈 Trade_bot dashboard")
st.caption(f"{cfg['symbol']} · {cfg['timeframe']} · trades on {cfg['exchange']} · "
           f"trained on {cfg['data_exchange']} data · "
           f"fee {cfg['fee']*100:.2f}% + slippage {cfg['slippage']*100:.2f}% per side")

tab_live, tab_train, tab_bt, tab_cfg = st.tabs(["🟢 Live (paper)", "Training progress",
                                                "Backtest", "Config"])


def line_chart(df, x, ys, title, yaxis, zero_line=False):
    fig = go.Figure()
    for col, name in ys:
        fig.add_trace(go.Scatter(x=df[x], y=df[col], name=name, mode="lines+markers"))
    if zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, yaxis_title=yaxis, xaxis_title=x, height=320,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


# ---------------- Live paper trading (auto-refreshing) ----------------
with tab_live:
    modes = [m for m in ("live", "replay") if (REPORT / f"paper_{m}" / "equity.csv").exists()]
    if not modes:
        st.info("No paper trading yet. Run `python paper_trade.py` (live Kraken prices) or "
                "`python paper_trade.py --replay` (fast demo on held-out data).")
    else:
        mode = st.radio("Mode", modes, horizontal=True,
                        format_func=lambda m: {"live": "Live prices", "replay": "Replay (demo)"}[m])

        @st.fragment(run_every="5s")
        def live_section(mode):
            folder = REPORT / f"paper_{mode}"
            eq = pd.read_csv(folder / "equity.csv")
            if eq.empty:
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

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                                vertical_spacing=0.06,
                                subplot_titles=("Price & trades", "Fake balance ($)"))
            fig.add_trace(go.Scatter(x=eq["time"], y=eq["price"], name="price",
                                     line=dict(color="gray", width=1)), 1, 1)
            if len(trades):
                for side, color, sym in (("BUY", "#2ca02c", "triangle-up"),
                                         ("SELL", "#d62728", "triangle-down")):
                    t = trades[trades["side"] == side].tail(1000)
                    fig.add_trace(go.Scatter(x=t["time"], y=t["price"], mode="markers",
                                             name=side.lower(),
                                             marker=dict(symbol=sym, size=9, color=color)), 1, 1)
            fig.add_trace(go.Scatter(x=eq["time"], y=eq["equity"], name="bot"), 2, 1)
            fig.add_trace(go.Scatter(x=eq["time"], y=eq["buy_hold"], name="buy & hold",
                                     opacity=0.7), 2, 1)
            fig.add_hline(y=start, line_dash="dash", line_color="gray", row=2, col=1)
            fig.update_layout(height=650, margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation="h"), uirevision=mode)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Trade log")
            if len(trades):
                st.dataframe(trades.iloc[::-1], use_container_width=True, hide_index=True)
            else:
                st.caption("No trades yet - the bot is staying in cash.")

        live_section(mode)


# ---------------- Training progress (auto-refreshing) ----------------
with tab_train:
    @st.fragment(run_every="5s")
    def training_section():
        log = REPORT / "training_log.csv"
        if not log.exists():
            st.info("No training log yet. Run `python train.py` — this page updates live.")
            return
        df = pd.read_csv(log)
        if df.empty:
            st.info("Training started, waiting for the first validation checkpoint…")
            return

        last, best = df.iloc[-1], df.loc[df["val_return_%"].idxmax()]
        target = cfg["train"]["total_timesteps"]
        st.progress(min(1.0, last["timesteps"] / target),
                    text=f"{int(last['timesteps']):,} / {target:,} timesteps "
                         f"· {last['elapsed_s']/60:.1f} min elapsed")

        c = st.columns(4)
        c[0].metric("Latest val return", f"{last['val_return_%']:+.2f}%",
                    f"{last['val_return_%'] - df['val_return_%'].iloc[-2]:+.2f}" if len(df) > 1 else None)
        c[1].metric("Best val return (saved)", f"{best['val_return_%']:+.2f}%",
                    f"at {int(best['timesteps']):,} steps", delta_color="off")
        c[2].metric("Val trades", int(last["val_trades"]),
                    f"{int(last['val_trades'] - df['val_trades'].iloc[-2]):+d}" if len(df) > 1 else None,
                    delta_color="inverse")
        c[3].metric("Val Sharpe", f"{last['val_sharpe']:.2f}")

        l, r = st.columns(2)
        l.plotly_chart(line_chart(df, "timesteps", [("val_return_%", "val return %")],
                                  "Validation return", "%", zero_line=True), use_container_width=True)
        r.plotly_chart(line_chart(df, "timesteps", [("val_trades", "trades")],
                                  "Trades on validation (fewer = less fee drag)", "trades"),
                       use_container_width=True)
        l2, r2 = st.columns(2)
        l2.plotly_chart(line_chart(df, "timesteps", [("train_ep_reward", "mean episode reward")],
                                   "Training reward", "reward", zero_line=True), use_container_width=True)
        r2.plotly_chart(line_chart(df, "timesteps", [("val_max_drawdown_%", "max drawdown %")],
                                   "Validation max drawdown", "%"), use_container_width=True)
        with st.expander("Raw log"):
            st.dataframe(df, use_container_width=True, hide_index=True)

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

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.4, 0.15],
                            vertical_spacing=0.04,
                            subplot_titles=("Equity (start = 1.0)", "Price & trades", "Position"))
        fig.add_trace(go.Scatter(x=eq["time"], y=eq["agent"], name="PPO agent"), 1, 1)
        fig.add_trace(go.Scatter(x=eq["time"], y=eq["buy_hold"], name="Buy & hold", opacity=0.7), 1, 1)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=1)

        fig.add_trace(go.Scatter(x=eq["time"], y=eq["close"], name="price",
                                 line=dict(color="gray", width=1)), 2, 1)
        chg = eq["position"].diff().fillna(eq["position"])
        buys, sells = eq[chg > 0], eq[chg < 0]
        if len(buys) + len(sells) <= 2000:   # too many markers makes the chart unreadable
            fig.add_trace(go.Scatter(x=buys["time"], y=buys["close"], mode="markers", name="buy",
                                     marker=dict(symbol="triangle-up", size=8, color="#2ca02c")), 2, 1)
            fig.add_trace(go.Scatter(x=sells["time"], y=sells["close"], mode="markers", name="sell",
                                     marker=dict(symbol="triangle-down", size=8, color="#d62728")), 2, 1)
        fig.add_trace(go.Scatter(x=eq["time"], y=eq["position"], name="position", fill="tozeroy",
                                 line=dict(shape="hv", width=0.5), showlegend=False), 3, 1)
        fig.update_yaxes(tickvals=[0, 1], ticktext=["flat", "long"], row=3, col=1)
        fig.update_layout(height=800, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Metrics")
        st.dataframe(mt, use_container_width=True)
        st.caption(f"Time in market: {eq['position'].mean()*100:.1f}% of bars")


# ---------------- Config ----------------
with tab_cfg:
    st.json(cfg)
