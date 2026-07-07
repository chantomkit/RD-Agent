"""
agent_loop.report_app — the Streamlit research-report UI (ADR 0003).

`streamlit run agent_loop/report_app.py -- --root <dir>` (the `agent_loop.report_ui` launcher wires this).
A read-only viewer over the agent-driven loop's file state: browse every loop's report — the gate
decision, the selection-vs-holdout comparison, the backtest figures (incl. per-ticker entry/exit), and
the agent's discussion — and copy a "branch" command to fine-tune from any historic loop.

All heavy lifting lives in `agent_loop.report` (the unit-tested, qlib-free read layer); this file is only
presentation. No qlib, no Docker, no `.env`.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import streamlit as st

from agent_loop import report as R

st.set_page_config(layout="wide", page_title="Quant R&D Reports", page_icon="🔬",
                   initial_sidebar_state="expanded")

_VERDICT_STYLE = {
    "SOTA": ("🏆 SOTA (new champion)", "#2ca02c"),
    "promote": ("✅ Promoted", "#2ca02c"),
    "reject": ("⛔ Rejected", "#d62728"),
}


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=R.DEFAULT_ROOT)
    known, _ = ap.parse_known_args(sys.argv[1:])
    return known


@st.cache_data(show_spinner=False)
def _artifacts(workspace: str | None, mtime_key: float):
    """Load all host-readable artifacts for a workspace once (cache-keyed on the ret.pkl mtime)."""
    return {
        "equity": R.equity_frame(workspace),
        "signal": R.signal_frames(workspace),
        "long_short": R.long_short_series(workspace),
        "positions": R.positions_frame(workspace),
        "prices": R.prices_frame(workspace),
    }


def _load_artifacts(workspace: str | None) -> dict:
    if not workspace:
        return {k: None for k in ("equity", "signal", "long_short", "positions", "prices")}
    art = R._find_artifact(workspace, "ret.pkl")
    mtime = art.stat().st_mtime if art else 0.0
    return _artifacts(workspace, mtime)


def _fmt(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*100:+.2f}%" if pct else f"{v:+.4f}"


def _verdict_badge(rep: R.Report):
    label, color = _VERDICT_STYLE.get(rep.verdict, (rep.verdict, "#888"))
    st.markdown(
        f"<span style='background:{color};color:white;padding:3px 10px;border-radius:6px;"
        f"font-weight:600'>{label}</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------------
def setup_panel(rep: R.Report):
    su = R.backtest_setup(rep)
    cfg = su["config"]
    st.subheader("Backtest setup")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Data & model**")
        st.markdown(
            f"- Universe: `{cfg.get('market', '?')}` ({rep.project.upper()})\n"
            f"- Data region: `{cfg.get('region', '?')}`\n"
            f"- Model: {cfg.get('model', '?')}\n"
            f"- Feature base: {cfg.get('handler', '?')}")
    with c2:
        st.markdown("**Benchmark**")
        st.markdown(f"- `{cfg.get('benchmark', '?')}` — {cfg.get('benchmark_name', '?')}")
        st.caption("Every 'excess return' and Information Ratio number is measured **against this market "
                   "index**. Holding the index *is* the `buy_hold` baseline below — it is **not** a "
                   "buy-and-hold of the strategy.")
    with c3:
        st.markdown("**Strategy & cost**")
        st.markdown(
            f"- {cfg.get('strategy', '?')}: hold top {cfg.get('topk', '?')}, drop "
            f"{cfg.get('n_drop', '?')}/day, min-hold {cfg.get('hold_thresh', '?')}d\n"
            f"- Cost per side: open {cfg.get('open_cost', '?')}, close {cfg.get('close_cost', '?')}\n"
            f"- Deal price: {cfg.get('deal_price', '?')}")
    seg = su["segments"]
    st.markdown(
        f"**Date segments** — Train `{seg.get('train', '?')}` · Valid `{seg.get('valid', '?')}` · "
        f"**Selection (test)** `{seg.get('selection_test', '?')}` · "
        f"**Locked holdout (test)** `{seg.get('holdout_test', '?')}`")
    feats = su.get("features")
    line = f"qlib config: `{su.get('conf', '?')}`"
    if feats:
        shown = ", ".join(feats[:8]) + (" …" if len(feats) > 8 else "")
        line += f"  ·  Features ({len(feats)}): {shown}"
    st.caption(line)


def gate_panel(rep: R.Report):
    st.subheader("Gate decision")
    if not rep.gates:
        st.info("No guardrail decision recorded for this loop.")
        return
    cols = st.columns(len(rep.gates))
    for col, (gate, ok) in zip(cols, rep.gates.items()):
        col.markdown(f"**{'✅' if ok else '❌'}**  \n{R.GATE_LABELS.get(gate, gate)}")
    dsr = rep.dsr.get("value")
    n = rep.dsr.get("n_trials")
    bench = rep.dsr.get("benchmark_sr_ann")
    bits = []
    if dsr is not None:
        bits.append(f"**Deflated Sharpe:** {dsr:.3f}")
    if n is not None:
        bits.append(f"trials N={n}")
    if bench is not None:
        bits.append(f"benchmark SR(ann)={bench:.3f}")
    if bits:
        st.caption("  ·  ".join(bits))
    if rep.reasons:
        with st.expander("Guardrail reasons (verbatim)"):
            for r in rep.reasons:
                st.markdown(f"- {r}")


def comparison_table(rep: R.Report):
    st.subheader("Selection vs. locked holdout")
    rows = []
    for key, label in R.COMPARISON_METRICS:
        s, h = rep.selection_metrics.get(key), rep.holdout_metrics.get(key)
        pct = "return" in key or "drawdown" in key
        rows.append({"Metric": label, "Selection": _fmt(s, pct), "Holdout": _fmt(h, pct)})
    st.table(pd.DataFrame(rows).set_index("Metric"))
    gap = rep.overfit_gap
    if gap is not None:
        flag = "⚠️ positive gap — watch for overfitting" if gap > 0.01 else "ok (holdout holds up)"
        st.caption(f"Selection − holdout Rank IC gap: **{gap:+.4f}**  ({flag})")

    # Absolute vs benchmark — bridges the (excess) metrics above and the equity figures below, so a
    # positive excess in a down market (absolute loss) is no longer confusing.
    sel_r, hold_r = R.segment_returns(rep.selection_workspace), R.segment_returns(rep.holdout_workspace)
    if sel_r or hold_r:
        st.markdown("**Absolute vs. benchmark — cumulative return over each window (from the equity curve):**")

        def _c(r, k):
            return _fmt(r.get(k), pct=True) if r else "—"

        abs_rows = [("Strategy (after cost)", "strategy_net"),
                    ("Benchmark index", "benchmark"),
                    ("Excess (strategy − benchmark)", "excess_net")]
        st.table(pd.DataFrame(
            [{"": name, "Selection": _c(sel_r, k), "Holdout": _c(hold_r, k)} for name, k in abs_rows]
        ).set_index(""))
        st.caption("The metrics table above is **excess vs. benchmark**. This table shows the absolute "
                   "picture: a positive *Excess* while *Strategy* is negative means the book lost money "
                   "but still beat the index (a falling market).")

    panel = rep.baselines.get("panel") or {}
    if panel:
        st.markdown("**Trivial-baseline panel (SOTA floor, holdout net IR):**")
        bar_by = rep.baselines.get("bar_set_by")
        st.table(pd.DataFrame(
            [{"Baseline": k + (" ← bar" if k == bar_by else ""), "Holdout net IR": f"{v:+.3f}"}
             for k, v in panel.items()]).set_index("Baseline"))

    with st.expander("What do these metrics mean?"):
        for term, desc in R.METRIC_GLOSSARY:
            st.markdown(f"- **{term}** — {desc}")


def _plotly(fig, key: str):
    # Streamlit derives an element id from type+params, so structurally-identical charts (e.g. the same
    # figure in the selection and holdout tabs) collide unless we pass a unique key.
    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    for i, f in enumerate(figs):
        st.plotly_chart(f, width="stretch", key=f"{key}_{i}")


def figures_for(label: str, workspace: str | None):
    art = _load_artifacts(workspace)
    if art["equity"] is None and art["signal"] is None:
        st.info(f"No backtest artifacts found for the {label} run "
                "(workspace missing or not yet produced).")
        return
    pred, lab = art["signal"] if art["signal"] else (None, None)

    # Primary, clearly-labelled return chart: absolute strategy vs benchmark vs excess.
    _plotly(R.cumulative_return_figure(art["equity"]), key=f"cum_{label}")
    st.caption("**Green** = excess vs benchmark (what the metrics table measures). **Blue** = the "
               "strategy's *absolute* return; **grey** = the benchmark. In a down market blue/grey can be "
               "deeply negative while green (excess) stays positive.")

    c1, c2 = st.columns(2)
    with c1:
        _plotly(R.drawdown_figure(art["equity"]), key=f"dd_{label}")
        _plotly(R.ic_timeseries_figure(pred, lab), key=f"ic_{label}")
        _plotly(R.long_short_cum_figure(art["long_short"]), key=f"ls_{label}")
    with c2:
        _plotly(R.portfolio_value_figure(art["equity"]), key=f"pf_{label}")
        _plotly(R.quantile_return_figure(pred, lab), key=f"quant_{label}")
        _plotly(R.turnover_figure(art["equity"]), key=f"to_{label}")

    # Tier-2 — real positions + price with entry/exit marks (this run's own test window)
    if art["positions"] is not None and art["prices"] is not None:
        st.markdown("**Price with strategy entry/exit — pick a holding**")
        dts = pd.to_datetime(art["prices"]["datetime"])
        st.caption(f"Trades over the **{label}** window ({dts.min():%Y-%m-%d} → {dts.max():%Y-%m-%d}); "
                   "each tab shows the entries/exits for its own segment.")
        tickers = R.held_tickers(art["positions"])
        if tickers:
            tk = st.selectbox("Ticker", tickers, key=f"ticker_{label}")
            _plotly(R.price_with_trades_figure(art["positions"], art["prices"], tk),
                    key=f"price_{label}")
    else:
        st.caption(f"Per-ticker price + entry/exit for the **{label}** segment needs Tier-2 "
                   "`positions.parquet`/`prices.parquet` (emitted per-run by the US template's extraction "
                   "step). Not present for this run.")

    with st.expander("Detailed qlib standard report (all series — dense)"):
        _plotly(R.qlib_standard_report_figure(art["equity"]), key=f"qlib_{label}")


def branch_panel(rep: R.Report):
    st.subheader("Fine-tune from this loop")
    st.caption("Read-only viewer: copy this into the agent to branch a new loop (ADR 0003).")
    st.code(f'Run a new quant-rd-loop iteration branching from {rep.report_id} '
            f'(parent_loop={rep.loop}, "{rep.hypothesis}"): <your refinement idea>', language="text")


# --------------------------------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------------------------------
def loop_view(rep: R.Report):
    top = st.container()
    with top:
        left, right = st.columns([3, 1])
        with left:
            st.title(f"{rep.title or rep.hypothesis}")
            st.markdown(f"`{rep.report_id}` · **{rep.action}** · loop {rep.loop}"
                        + (f" · branched from loop {rep.parent_loop}" if rep.parent_loop is not None else "")
                        + (f" · {rep.timestamp}" if rep.timestamp else ""))
            st.markdown(f"**Hypothesis:** {rep.hypothesis}")
        with right:
            _verdict_badge(rep)
    st.divider()

    setup_panel(rep)
    st.divider()
    gate_panel(rep)
    st.divider()
    comparison_table(rep)
    st.divider()

    st.subheader("Backtest figures")
    tabs = st.tabs(["Selection segment", "Locked holdout"])
    with tabs[0]:
        figures_for("selection", rep.selection_workspace)
    with tabs[1]:
        figures_for("holdout", rep.holdout_workspace)
    st.divider()

    st.subheader("Discussion")
    if rep.report_md:
        st.markdown(rep.report_md)
    else:
        st.info("No discussion written yet for this loop (predates the report.md step, or not authored).")
    st.divider()
    branch_panel(rep)


def trends_view(reps: list[R.Report]):
    st.title("Project trends across loops")
    if not reps:
        st.info("No loops recorded.")
        return
    df = pd.DataFrame([{
        "loop": r.loop,
        "sel Rank IC": r.selection_metrics.get("Rank IC"),
        "hold Rank IC": r.holdout_metrics.get("Rank IC"),
        "hold net IR": r.holdout_metrics.get(R.NET_IR),
        "DSR": r.dsr.get("value"),
        "overfit gap": r.overfit_gap,
    } for r in reps]).set_index("loop")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rank IC: selection vs holdout**")
        st.line_chart(df[["sel Rank IC", "hold Rank IC"]])
        st.markdown("**Deflated Sharpe by loop**")
        st.line_chart(df[["DSR"]])
    with c2:
        st.markdown("**Holdout net IR by loop**")
        st.line_chart(df[["hold net IR"]])
        st.markdown("**Overfitting gap (sel − hold Rank IC)**")
        st.line_chart(df[["overfit gap"]])
    st.dataframe(df, width="stretch")


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
def main():
    args = _args()
    st.sidebar.title("🔬 Quant R&D Reports")
    ledgers = R.discover_ledgers(args.root)
    if not ledgers:
        st.warning(f"No `trace*.json` ledgers found under `{args.root}`. "
                   "Run a loop iteration first (see the quant-rd-loop skill).")
        return

    labels = {str(p): f"{R.project_name(p)}  ({len(R.load_ledger(p).get('trials', []))} loops)"
              for p in ledgers}
    ledger_path = st.sidebar.selectbox("Project (ledger)", [str(p) for p in ledgers],
                                       format_func=lambda p: labels[p])
    ledger = R.load_ledger(ledger_path)
    reps = R.load_all(ledger_path)

    sota = ledger.get("sota_loop")
    base = ledger.get("baselines") or {}
    if base.get("bar") is not None:
        st.sidebar.caption(f"SOTA floor: {base['bar']:+.3f} ({base.get('bar_set_by')})")
    st.sidebar.caption(f"Champion: {'loop ' + str(sota) if sota is not None else 'none (floor unbeaten)'}")

    view = st.sidebar.radio("View", ["Loop detail", "Project trends"])
    if view == "Project trends":
        trends_view(reps)
        return

    if not reps:
        st.info("This ledger has no loops yet.")
        return

    def _loop_label(r: R.Report) -> str:
        mark = {"SOTA": "🏆", "promote": "✅", "reject": "⛔"}.get(r.verdict, "•")
        return f"{mark} loop {r.loop} — {(r.title or r.hypothesis)[:48]}"

    by_loop = {r.loop: r for r in reps}
    chosen = st.sidebar.radio("Loop", [r.loop for r in reps], format_func=lambda n: _loop_label(by_loop[n]))
    loop_view(by_loop[chosen])


main()
