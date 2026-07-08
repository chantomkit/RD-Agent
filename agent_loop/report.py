"""
agent_loop.report — the read layer for the research-report UI (ADR 0003).

A pure, host-side reader over the file-based state the agent-driven loop (ADR 0002) already produces: the
plain-JSON trace ledger, per-loop run/decision JSON, the agent's `report.md` narrative, and the qlib
workspace artifacts. It assembles a `Report` per loop and builds the backtest figures.

Design constraints:
  * NO qlib on the host (ADR 0002 keeps qlib in Docker). Every artifact this module loads is plain
    pandas — `ret.pkl` (portfolio report), `pred.pkl`/`label.pkl` (signal + forward return),
    `long_short_r.pkl`, and the Tier-2 `positions.parquet`/`prices.parquet` that the in-container
    extraction (us_template/read_exp_res.py) dumps. Files needing qlib (`positions_normal_1day.pkl`,
    the binary price data) are never touched here.
  * NO Streamlit import — so the loader + figure builders are unit-testable without a UI.
  * Everything degrades gracefully: a missing artifact yields None / an empty figure, never a crash.

The one qlib helper we reuse is `rdagent.log.ui.qlib_report_figure.report_figure`, imported lazily
inside `qlib_standard_report_figure` (it is pure pandas/plotly at call time — verified on the host).
"""
from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

DEFAULT_ROOT = "git_ignore_folder/agent_loop"

# Metrics surfaced in the selection-vs-holdout comparison table, in display order.
# Labels spell out the jargon; the return/IR metrics are EXCESS over the benchmark (see glossary).
COMPARISON_METRICS = [
    ("Rank IC", "Rank Information Coefficient (Rank IC)"),
    ("IC", "Information Coefficient (IC)"),
    ("ICIR", "IC Information Ratio (ICIR)"),
    ("Rank ICIR", "Rank-IC Information Ratio (Rank ICIR)"),
    ("1day.excess_return_with_cost.information_ratio", "Information Ratio — excess vs benchmark, after cost (IR)"),
    ("1day.excess_return_with_cost.annualized_return", "Annualized return — excess vs benchmark, after cost"),
    ("1day.excess_return_with_cost.max_drawdown", "Max drawdown — of the excess-return curve"),
    ("Long-Short Ann Sharpe", "Long-Short Sharpe — market-neutral (L/S)"),
]
NET_IR = "1day.excess_return_with_cost.information_ratio"

# Plain-language definitions rendered in the UI glossary.
METRIC_GLOSSARY = [
    ("Information Coefficient (IC)", "Correlation between the model's predicted scores and the returns "
     "that actually followed. Higher = the ranking predicts returns better. ~0.03 is already useful."),
    ("Rank IC", "Same as IC but on ranks (Spearman) — robust to outliers. The headline signal-quality number."),
    ("ICIR / Rank ICIR", "IC divided by its own volatility across days — how *consistent* the signal is, "
     "not just how strong on average."),
    ("Information Ratio (IR)", "Excess return over the benchmark ÷ the volatility of that excess "
     "(tracking error). Risk-adjusted outperformance. Here it is net of trading cost."),
    ("Excess vs benchmark", "Strategy return minus the benchmark index return. IMPORTANT: a positive "
     "excess in a falling market still means the account lost money — you just lost less than the index. "
     "See the 'Absolute vs benchmark' table."),
    ("Long-Short Sharpe (market-neutral)", "Sharpe of a book that is long the top-ranked names and short "
     "the bottom-ranked — market beta removed, so it isolates pure signal quality (alpha)."),
    ("Deflated Sharpe Ratio (DSR)", "The Sharpe ratio haircut for multiple testing and short samples — "
     "the bar rises as you try more ideas, guarding against luck."),
    ("Max drawdown", "The worst peak-to-trough decline over the window (a loss measure)."),
]

# Human-readable gate names for the decision panel.
GATE_LABELS = {
    "holdout_ok": "Beats SOTA on locked holdout",
    "dsr_ok": "Deflated Sharpe ≥ threshold",
    "net_positive": "Net-of-cost IR > 0",
    "beats_sota_net": "Net IR beats incumbent",
    "beats_baselines": "Beats trivial-baseline panel",
    "neutral_alpha_ok": "Market-neutral alpha (L/S) > 0",
}


# --------------------------------------------------------------------------------------------------
# Ledger / project discovery
# --------------------------------------------------------------------------------------------------
def discover_ledgers(root: str = DEFAULT_ROOT) -> list[Path]:
    """All `trace*.json` ledgers under `root` (each ledger = one research project, e.g. cn/us)."""
    r = Path(root)
    if not r.is_dir():
        return []
    return sorted(p for p in r.glob("trace*.json") if p.is_file())


def project_name(ledger_path: str | Path) -> str:
    """A short project id from the ledger filename: trace_us.json -> 'us', trace.json -> 'cn'."""
    stem = Path(ledger_path).stem  # 'trace' or 'trace_us'
    suffix = stem[len("trace"):].lstrip("_")
    return suffix or "cn"


def load_ledger(ledger_path: str | Path) -> dict[str, Any]:
    p = Path(ledger_path)
    if not p.exists():
        return {"trials": [], "sota_loop": None}
    return json.loads(p.read_text())


def run_dir_for(ledger_path: str | Path, loop: int, trial: dict | None = None) -> Path:
    """Locate a loop's run directory. Explicit `run_dir` on the trial wins; else convention from the
    ledger stem: trace_us.json -> runs_us/loop_N, trace.json -> runs/loop_N."""
    if trial and trial.get("run_dir"):
        return Path(trial["run_dir"])
    ledger = Path(ledger_path)
    proj = project_name(ledger)
    runs = "runs" if proj == "cn" else f"runs_{proj}"
    return ledger.parent / runs / f"loop_{loop}"


# --------------------------------------------------------------------------------------------------
# The Report model
# --------------------------------------------------------------------------------------------------
@dataclass
class Report:
    project: str
    loop: int
    action: str
    hypothesis: str
    verdict: str  # "SOTA" | "promote" | "reject"
    is_sota: bool
    title: str | None = None
    parent_loop: int | None = None
    timestamp: str | None = None

    selection_segment: dict | None = None
    holdout_segment: dict | None = None
    selection_metrics: dict = field(default_factory=dict)
    holdout_metrics: dict = field(default_factory=dict)

    gates: dict = field(default_factory=dict)
    dsr: dict = field(default_factory=dict)  # {value, n_trials, benchmark_sr_ann, ...}
    reasons: list[str] = field(default_factory=list)
    baselines: dict = field(default_factory=dict)  # {panel, bar, bar_set_by}

    selection_workspace: str | None = None
    holdout_workspace: str | None = None
    report_md: str | None = None
    run_dir: str | None = None
    conf: str | None = None          # qlib config file used (e.g. conf_baseline.yaml)
    settings: dict = field(default_factory=dict)  # explicit run_qlib knobs (topk/costs/…), if recorded

    @property
    def report_id(self) -> str:
        return f"{self.project}/{self.loop}"

    @property
    def overfit_gap(self) -> float | None:
        s, h = self.selection_metrics.get("Rank IC"), self.holdout_metrics.get("Rank IC")
        return (s - h) if (s is not None and h is not None) else None


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_report(ledger_path: str | Path, loop: int, ledger: dict | None = None) -> Report:
    """Assemble a Report for one loop from the ledger trial + run/decision JSON + report.md."""
    ledger = ledger if ledger is not None else load_ledger(ledger_path)
    trials = ledger.get("trials", [])
    trial = next((t for t in trials if t.get("loop") == loop), None)
    if trial is None:
        raise KeyError(f"loop {loop} not in ledger {ledger_path}")

    sota = ledger.get("sota_loop")
    is_sota = sota == loop
    verdict = "SOTA" if is_sota else ("promote" if trial.get("decision") else "reject")

    rundir = run_dir_for(ledger_path, loop, trial)
    sel = _read_json(rundir / "sel.json") or {}
    hold = _read_json(rundir / "hold.json") or {}
    decision = _read_json(rundir / "decision.json") or {}

    report_md = None
    md_path = Path(trial["report_path"]) if trial.get("report_path") else (rundir / "report.md")
    if md_path.exists():
        report_md = md_path.read_text()

    # Prefer the full metric set from the run JSON; fall back to the compact set stored on the trial.
    sel_metrics = sel.get("metrics") or trial.get("selection_metrics") or {}
    hold_metrics = hold.get("metrics") or trial.get("holdout_metrics") or {}

    dsr = decision.get("dsr") or {}
    if not dsr and trial.get("guardrail"):
        dsr = {"value": trial["guardrail"].get("dsr")}

    base = ledger.get("baselines") or {}
    baselines = {
        "panel": base.get("panel") or decision.get("baselines"),
        "bar": base.get("bar"),
        "bar_set_by": base.get("bar_set_by"),
    }

    return Report(
        project=project_name(ledger_path),
        loop=loop,
        action=trial.get("action", "factor"),
        hypothesis=trial.get("hypothesis", ""),
        verdict=verdict,
        is_sota=is_sota,
        title=trial.get("title"),
        parent_loop=trial.get("parent_loop"),
        timestamp=trial.get("timestamp"),
        selection_segment=trial.get("selection_segment") or sel.get("segments"),
        holdout_segment=trial.get("holdout_segment") or hold.get("segments"),
        selection_metrics=sel_metrics,
        holdout_metrics=hold_metrics,
        gates=(decision.get("gates") or (trial.get("guardrail") or {}).get("gates") or {}),
        dsr=dsr,
        reasons=decision.get("reasons") or [],
        baselines=baselines,
        selection_workspace=sel.get("workspace") or trial.get("candidate_workspace"),
        holdout_workspace=hold.get("workspace"),
        report_md=report_md,
        run_dir=str(rundir),
        conf=sel.get("conf") or hold.get("conf"),
        settings=sel.get("settings") or {},
    )


# --------------------------------------------------------------------------------------------------
# Backtest setup (metadata) — parsed from the run's qlib config so the UI can show the whole setup
# --------------------------------------------------------------------------------------------------
BENCHMARK_NAMES = {
    "SPX": "S&P 500 index (cap-weighted)",
    "SH000300": "CSI 300 index",
    "SH000905": "CSI 500 index",
    "SH000016": "SSE 50 index",
}


def _conf_val(text: str, key: str) -> str | None:
    """Read a scalar from a qlib config, handling `key: &anchor VALUE`, `key: VALUE`, and the jinja
    `key: {{ key | default(VALUE, true) }}` forms (US templates use jinja; CN uses concrete values)."""
    m = re.search(rf"^\s*{re.escape(key)}:\s*\{{\{{[^}}]*?default\(\s*([^,)\s]+)", text, re.M)
    if m:
        return m.group(1)
    m = re.search(rf"^\s*{re.escape(key)}:\s*(?:&\S+\s+)?([^\s#]+)", text, re.M)
    return m.group(1) if m else None


def _parse_conf(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    model_m = re.search(r"class:\s*(\w*Model)\b", text)
    label_m = re.search(r'label:\s*\n\s*-\s*\[\s*"([^"]+)"', text)
    out = {
        "market": _conf_val(text, "market"),
        "benchmark": _conf_val(text, "benchmark"),
        "region": _conf_val(text, "region"),
        "topk": _conf_val(text, "topk"),
        "n_drop": _conf_val(text, "n_drop"),
        "hold_thresh": _conf_val(text, "hold_thresh"),
        "open_cost": _conf_val(text, "open_cost"),
        "close_cost": _conf_val(text, "close_cost"),
        "deal_price": _conf_val(text, "deal_price"),
        "handler": "Alpha360" if "Alpha360" in text else ("Alpha158" if "Alpha158" in text else None),
        "model": ("LightGBM (LGBModel)" if "LGBModel" in text else (model_m.group(1) if model_m else None)),
        "strategy": "TopkDropout" if "TopkDropout" in text else None,
        "label": label_m.group(1) if label_m else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _seg_str(seg: dict | None, kind: str) -> str | None:
    seg = seg or {}
    a, b = seg.get(f"{kind}_start"), seg.get(f"{kind}_end")
    return f"{a} → {b}" if a and b else None


def backtest_setup(rep: "Report") -> dict[str, Any]:
    """Assemble the backtest setup shown at the top of a report: parsed qlib config (universe, benchmark,
    strategy, cost, model), the date segments, and any added features. Explicit run_qlib knobs override
    the config defaults; missing pieces are simply omitted."""
    conf_text = None
    if rep.selection_workspace and rep.conf:
        p = Path(rep.selection_workspace) / rep.conf
        if p.exists():
            conf_text = p.read_text()
    cfg = _parse_conf(conf_text)
    for k in ("topk", "n_drop", "hold_thresh", "open_cost", "close_cost"):
        if rep.settings.get(k) is not None:
            cfg[k] = rep.settings[k]
    cfg["benchmark_name"] = BENCHMARK_NAMES.get(cfg.get("benchmark"), cfg.get("benchmark"))

    feats = None
    if rep.run_dir and (Path(rep.run_dir) / "features.json").exists():
        try:
            feats = list(json.loads((Path(rep.run_dir) / "features.json").read_text()).keys())
        except (json.JSONDecodeError, OSError):
            feats = None

    return {
        "conf": rep.conf,
        "config": cfg,
        "features": feats,
        "segments": {
            "train": _seg_str(rep.selection_segment, "train"),
            "valid": _seg_str(rep.selection_segment, "valid"),
            "selection_test": _seg_str(rep.selection_segment, "test"),
            "holdout_test": _seg_str(rep.holdout_segment, "test"),
        },
    }


def _int(v: Any, default: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _label_description(expr: str | None) -> str:
    if not expr:
        return "each stock's next-period forward return"
    if expr.replace(" ", "") == "Ref($close,-2)/Ref($close,-1)-1":
        return ("each stock's **next-day forward return** (close-to-close, offset by one day so the trade "
                "is actually executable — the model never sees same-day information)")
    return f"a forward return defined by `{expr}`"


def strategy_explainer(rep: "Report") -> list[tuple[str, str]]:
    """A plain-English, per-report walk-through of the strategy mechanism — what the model predicts and
    exactly how positions are entered and exited — generated from the run's qlib config so it is always
    present and consistent (the agent's `report.md` carries the *specific* rationale on top of this)."""
    cfg = backtest_setup(rep)["config"]
    feats = backtest_setup(rep)["features"]
    topk, n_drop = _int(cfg.get("topk")), _int(cfg.get("n_drop"))
    hold = cfg.get("hold_thresh", "1")
    model = cfg.get("model", "a gradient-boosted tree model")
    handler = cfg.get("handler", "the factor")
    market = cfg.get("market", "the universe")
    bench = cfg.get("benchmark_name") or cfg.get("benchmark") or "the benchmark index"
    label_desc = _label_description(cfg.get("label"))
    feat_clause = (f"{len(feats)} factor features (the {handler} base set plus this loop's added factors)"
                   if feats else f"the {handler} factor set")
    turnover = f"about {n_drop}/{topk} ≈ {round(100 * n_drop / topk)}%" if (topk and n_drop) else "a small fraction"

    return [
        ("What the model predicts",
         f"A **{model}** is trained on the *train* window to predict {label_desc}, from {feat_clause}. "
         "Feature values are cross-sectionally normalized (outliers clipped) first. On every trading day "
         "of the test window the model scores every stock — a higher score means it is predicted to "
         "outperform its peers."),
        ("Entry — how positions are opened",
         f"Each day all **{market}** stocks are ranked by that score. The book targets the **top {topk}** "
         f"names, roughly equal-weighted (~{round(100 / topk)}% each if topk={topk}). A stock is "
         "**bought (▲ on the price charts)** when it climbs into the top ranks and a slot is available."),
        ("Exit — how positions are closed",
         f"This is a **Top-{topk} Dropout** rule: each day the **{n_drop} worst-scored names you currently "
         f"hold** are **sold (▼)** and replaced by the highest-scored names you don't yet hold (a name must "
         f"be held ≥ {hold} day(s) before it can be dropped). So a position exits when its rank *falls* far "
         "enough to be among the daily drops — there is **no fixed price target or stop-loss**. This caps "
         f"turnover at {turnover} of the book per day."),
        ("Execution & cost",
         f"Orders fill at the **{cfg.get('deal_price', 'close')} price**, with **{cfg.get('open_cost', '?')}** "
         f"buy / **{cfg.get('close_cost', '?')}** sell cost per side."),
        ("How it is scored",
         f"Results are reported as **excess return over {bench}**. Because a long-only top-k book is always "
         "net-long the market, a dollar-neutral **long-short Sharpe** (long the top names, short the bottom) "
         "is also computed to separate genuine stock-selection skill from simply riding market beta."),
        ("This iteration's idea",
         rep.hypothesis or "(no hypothesis recorded for this loop)"),
    ]


def load_all(ledger_path: str | Path) -> list[Report]:
    ledger = load_ledger(ledger_path)
    out = []
    for t in ledger.get("trials", []):
        try:
            out.append(load_report(ledger_path, t["loop"], ledger=ledger))
        except KeyError:
            continue
    return out


# --------------------------------------------------------------------------------------------------
# Workspace artifact loaders (all plain pandas; no qlib; tolerate absence)
# --------------------------------------------------------------------------------------------------
def _find_artifact(workspace: str | None, relpath: str) -> Path | None:
    """Find an artifact in a qlib workspace: top-level first, then the mlflow recorder tree. Returns the
    most-recently-modified match, or None."""
    if not workspace:
        return None
    ws = Path(workspace)
    top = ws / relpath
    if top.exists():
        return top
    hits = [Path(p) for p in glob.glob(str(ws / "mlruns" / "*" / "*" / "artifacts" / relpath))]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def equity_frame(workspace: str | None) -> pd.DataFrame | None:
    """The portfolio report (ret.pkl): daily return/cost/turnover/bench/account. Feeds report_figure."""
    p = _find_artifact(workspace, "ret.pkl") or _find_artifact(
        workspace, "portfolio_analysis/report_normal_1day.pkl"
    )
    return pd.read_pickle(p) if p else None


def signal_frames(workspace: str | None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(pred, label): MultiIndex [datetime, instrument] score + forward return."""
    pp, lp = _find_artifact(workspace, "pred.pkl"), _find_artifact(workspace, "label.pkl")
    if pp is None or lp is None:
        return None
    return pd.read_pickle(pp), pd.read_pickle(lp)


def long_short_series(workspace: str | None) -> pd.Series | None:
    p = _find_artifact(workspace, "sig_analysis/long_short_r.pkl")
    return pd.read_pickle(p) if p else None


def positions_frame(workspace: str | None) -> pd.DataFrame | None:
    """Tier-2 host-readable positions (us_template extraction). [datetime, instrument] -> amount/weight."""
    p = _find_artifact(workspace, "positions.parquet")
    return pd.read_parquet(p) if p else None


def prices_frame(workspace: str | None) -> pd.DataFrame | None:
    """Tier-2 host-readable prices (us_template extraction). [datetime, instrument] -> close."""
    p = _find_artifact(workspace, "prices.parquet")
    return pd.read_parquet(p) if p else None


# --------------------------------------------------------------------------------------------------
# Figure builders (pure plotly; return a go.Figure)
# --------------------------------------------------------------------------------------------------
def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=200)
    return fig


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(0.0, index=df.index)


def cumulative_return_figure(ret_df: pd.DataFrame | None):
    """Clean, labelled cumulative-return chart. Shows ABSOLUTE strategy return, the benchmark, and the
    EXCESS (strategy − benchmark) on the same axes — so a positive excess in a falling market (absolute
    loss) is visible at a glance rather than confusing."""
    if ret_df is None or ret_df.empty:
        return _empty("no ret.pkl for this run")
    net = (_col(ret_df, "return") - _col(ret_df, "cost")).cumsum()
    gross = _col(ret_df, "return").cumsum()
    bench = _col(ret_df, "bench").cumsum()
    excess = (_col(ret_df, "return") - _col(ret_df, "cost") - _col(ret_df, "bench")).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net.index, y=net.values, mode="lines",
                             name=f"Strategy, after cost (ends {net.iloc[-1]:+.1%})",
                             line=dict(color="#5b8ff9", width=2)))
    fig.add_trace(go.Scatter(x=gross.index, y=gross.values, mode="lines",
                             name="Strategy, before cost", line=dict(color="#9fc4fb", dash="dot")))
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, mode="lines",
                             name=f"Benchmark index (ends {bench.iloc[-1]:+.1%})",
                             line=dict(color="#8c8c8c", dash="dash")))
    fig.add_trace(go.Scatter(x=excess.index, y=excess.values, mode="lines",
                             name=f"Excess vs benchmark (ends {excess.iloc[-1]:+.1%})",
                             line=dict(color="#2ca02c", width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title="Cumulative return over the backtest",
                      xaxis_title="Date", yaxis_title="Cumulative return (sum of daily)",
                      yaxis_tickformat=".0%", height=400, legend_orientation="h",
                      legend_y=-0.2, margin=dict(t=40, b=20))
    return fig


def drawdown_figure(ret_df: pd.DataFrame | None):
    """Underwater (drawdown) curve — worst peak-to-trough decline over time — for the absolute strategy
    and the excess-vs-benchmark curves."""
    if ret_df is None or ret_df.empty:
        return _empty("no ret.pkl for this run")
    net_cum = (_col(ret_df, "return") - _col(ret_df, "cost")).cumsum()
    exc_cum = (_col(ret_df, "return") - _col(ret_df, "cost") - _col(ret_df, "bench")).cumsum()
    fig = go.Figure()
    for cum, name, color in [(net_cum, "Strategy (absolute, after cost)", "#d62728"),
                             (exc_cum, "Excess vs benchmark", "#2ca02c")]:
        dd = cum - cum.cummax()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name,
                                 fill="tozeroy", line=dict(color=color)))
    fig.update_layout(title="Drawdown (underwater curve) — how far below the prior peak",
                      xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%",
                      height=300, legend_orientation="h", legend_y=-0.25, margin=dict(t=40, b=20))
    return fig


def turnover_figure(ret_df: pd.DataFrame | None):
    if ret_df is None or ret_df.empty or "turnover" not in ret_df.columns:
        return _empty("no turnover series for this run")
    fig = go.Figure(go.Bar(x=ret_df.index, y=ret_df["turnover"], marker_color="#73aaf5"))
    fig.update_layout(title="Daily turnover (fraction of the book traded)",
                      xaxis_title="Date", yaxis_title="Turnover", yaxis_tickformat=".0%",
                      height=260, margin=dict(t=40, b=20))
    return fig


def qlib_standard_report_figure(ret_df: pd.DataFrame | None):
    """The reused qlib multi-panel report (all series). Rich but dense — shown in a collapsed expander."""
    if ret_df is None or ret_df.empty:
        return _empty("no ret.pkl for this run")
    from rdagent.log.ui.qlib_report_figure import report_figure

    fig = report_figure(ret_df)
    fig.update_layout(title="qlib standard backtest report (all cumulative / drawdown / turnover series)")
    return fig


def segment_returns(workspace: str | None) -> dict | None:
    """Cumulative net returns over a run's window, from ret.pkl — for the 'absolute vs benchmark' table
    that bridges the (excess) metrics and the equity figures."""
    df = equity_frame(workspace)
    if df is None or df.empty:
        return None
    strat = float((_col(df, "return") - _col(df, "cost")).sum())
    bench = float(_col(df, "bench").sum())
    acct = None
    if "account" in df.columns and len(df) > 1 and df["account"].iloc[0]:
        acct = float(df["account"].iloc[-1] / df["account"].iloc[0] - 1)
    return {"strategy_net": strat, "benchmark": bench, "excess_net": strat - bench, "account_change": acct}


def _first_col(df: pd.DataFrame) -> pd.Series:
    return df.iloc[:, 0]


def daily_ic(pred: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional IC (pearson) and Rank IC (spearman) of score vs forward return."""
    s = _first_col(pred).rename("score")
    y = _first_col(label).rename("label")
    df = pd.concat([s, y], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=["IC", "Rank IC"])
    g = df.groupby(level="datetime")
    ic = g.apply(lambda d: d["score"].corr(d["label"]))
    ric = g.apply(lambda d: d["score"].corr(d["label"], method="spearman"))
    return pd.DataFrame({"IC": ic, "Rank IC": ric}).dropna()


def ic_timeseries_figure(pred: pd.DataFrame | None, label: pd.DataFrame | None):
    if pred is None or label is None:
        return _empty("no pred/label for this run")
    ic = daily_ic(pred, label)
    if ic.empty:
        return _empty("IC series empty (labels all NaN?)")
    fig = go.Figure()
    for col in ("IC", "Rank IC"):
        fig.add_trace(go.Scatter(x=ic.index, y=ic[col].rolling(5, min_periods=1).mean(),
                                 mode="lines", name=f"{col} (5d MA)"))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    mean_ric = ic["Rank IC"].mean()
    fig.update_layout(title=f"Daily signal quality — cross-sectional IC (mean Rank IC {mean_ric:+.4f})",
                      xaxis_title="Date", yaxis_title="Cross-sectional correlation (score vs realized return)",
                      height=320, legend_orientation="h", legend_y=-0.25, margin=dict(t=40, b=20))
    return fig


def quantile_return_figure(pred: pd.DataFrame | None, label: pd.DataFrame | None, q: int = 10):
    """Mean forward return by prediction quantile — the signal's monotonicity."""
    if pred is None or label is None:
        return _empty("no pred/label for this run")
    s = _first_col(pred).rename("score")
    y = _first_col(label).rename("label")
    df = pd.concat([s, y], axis=1).dropna()
    if df.empty:
        return _empty("signal/label empty")

    def _bucket(d):
        try:
            return pd.qcut(d["score"].rank(method="first"), q, labels=False)
        except ValueError:
            return pd.Series(index=d.index, dtype=float)

    df["bucket"] = df.groupby(level="datetime", group_keys=False).apply(_bucket)
    means = df.dropna(subset=["bucket"]).groupby("bucket")["label"].mean()
    if means.empty:
        return _empty("not enough names to bucket")
    fig = go.Figure(go.Bar(x=[f"Q{int(b)+1}" for b in means.index], y=means.values,
                           marker_color="#5b8ff9"))
    fig.update_layout(title=f"Signal monotonicity — mean next-period return by score quantile "
                            f"(Q1 = lowest score … Q{q} = highest)",
                      xaxis_title="Predicted-score quantile", yaxis_title="Mean next-period return",
                      yaxis_tickformat=".2%", height=320, margin=dict(t=50, b=20))
    return fig


def long_short_cum_figure(ls: pd.Series | None):
    if ls is None or len(ls) == 0:
        return _empty("no long_short_r.pkl for this run")
    cum = ls.cumsum()
    fig = go.Figure(go.Scatter(x=cum.index, y=cum.values, mode="lines", name="cumulative L/S return",
                               line=dict(color="#2ca02c")))
    fig.update_layout(title=f"Market-neutral long-short return (long top / short bottom; ends "
                            f"{cum.iloc[-1]:+.1%})",
                      xaxis_title="Date", yaxis_title="Cumulative long-short return",
                      yaxis_tickformat=".0%", height=320, margin=dict(t=40, b=20))
    return fig


def portfolio_value_figure(ret_df: pd.DataFrame | None):
    """ABSOLUTE account value (mark-to-market) — this is the real money in the book, before any
    comparison to the benchmark. It falls in a down market even when the excess return is positive."""
    if ret_df is None or ret_df.empty or "account" not in ret_df.columns:
        return _empty("no account series for this run")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ret_df.index, y=ret_df["account"], mode="lines", name="account value",
                             line=dict(color="#5b8ff9")))
    if "cash" in ret_df.columns:
        fig.add_trace(go.Scatter(x=ret_df.index, y=ret_df["cash"], mode="lines", name="cash",
                                 line=dict(color="#b0b0b0")))
    fig.update_layout(title="Absolute account value over time (mark-to-market, not vs benchmark)",
                      xaxis_title="Date", yaxis_title="Account value ($)",
                      height=300, legend_orientation="h", legend_y=-0.25, margin=dict(t=40, b=20))
    return fig


def _held_intervals(positions: pd.DataFrame, ticker: str) -> tuple[list, list]:
    """Entry/exit dates for a ticker: entry = a date it is held while the prior date it was not; exit =
    the first date after a held run where it is no longer held."""
    held = positions[positions["instrument"] == ticker] if "instrument" in positions.columns \
        else positions.xs(ticker, level="instrument")
    dates = sorted(pd.to_datetime(held["datetime"]).unique()) if "datetime" in positions.columns \
        else sorted(pd.to_datetime(held.index.get_level_values("datetime")).unique())
    if not dates:
        return [], []
    entries, exits, dset = [], [], set(dates)
    all_dates = sorted(pd.to_datetime(positions["datetime"]).unique()) if "datetime" in positions.columns \
        else sorted(pd.to_datetime(positions.index.get_level_values("datetime")).unique())
    idx = {d: i for i, d in enumerate(all_dates)}
    for d in dates:
        i = idx[d]
        if i == 0 or all_dates[i - 1] not in dset:
            entries.append(d)
    for d in dates:
        i = idx[d]
        if i + 1 >= len(all_dates) or all_dates[i + 1] not in dset:
            exits.append(all_dates[min(i + 1, len(all_dates) - 1)])
    return entries, exits


def price_with_trades_figure(positions: pd.DataFrame | None, prices: pd.DataFrame | None, ticker: str):
    """Tier-2: a ticker's price line with ▲ entry / ▼ exit markers from actual positions."""
    if positions is None or prices is None:
        return _empty("Tier-2 positions/prices not available for this run")
    px = prices[prices["instrument"] == ticker] if "instrument" in prices.columns \
        else prices.xs(ticker, level="instrument")
    if px is None or len(px) == 0:
        return _empty(f"no price series for {ticker}")
    px = px.copy()
    if "datetime" in px.columns:
        px = px.set_index("datetime")
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    close = px["close"] if "close" in px.columns else _first_col(px)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name=f"{ticker} close",
                             line=dict(color="#5b8ff9")))
    entries, exits = _held_intervals(positions, ticker)
    for dates, sym, color, name in [(entries, "triangle-up", "#2ca02c", "entry"),
                                    (exits, "triangle-down", "#d62728", "exit")]:
        dd = [d for d in dates if d in close.index]
        if dd:
            fig.add_trace(go.Scatter(x=dd, y=[close.loc[d] for d in dd], mode="markers", name=name,
                                     marker=dict(symbol=sym, size=11, color=color)))
    fig.update_layout(title=f"{ticker} — adjusted close with strategy entry (▲) / exit (▼)",
                      xaxis_title="Date", yaxis_title="Adjusted close price ($)",
                      height=360, legend_orientation="h", legend_y=-0.2, margin=dict(t=40, b=20))
    return fig


def held_tickers(positions: pd.DataFrame | None) -> list[str]:
    if positions is None:
        return []
    col = positions["instrument"] if "instrument" in positions.columns \
        else positions.index.get_level_values("instrument")
    return sorted(pd.Index(col).unique().tolist())
