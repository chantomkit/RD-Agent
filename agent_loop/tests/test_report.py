"""
Tests for the research-report read layer (ADR 0003, agent_loop/report.py).

Self-contained: fabricates a ledger + run/decision JSON + workspace artifacts (ret/pred/label/long_short
and Tier-2 positions/prices) in tmp_path. Exercises Report assembly, graceful degradation, the IC/quantile
figure builders, and the Tier-2 price-with-entry/exit plot. No qlib, no Docker, no Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from agent_loop import report as R


# --------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------
def _make_workspace(root: Path, name: str, rank_ic: float = 0.10) -> Path:
    """A qlib-like workspace with plain-pandas artifacts at the top level (equity + signal + L/S)."""
    ws = root / "ws" / name
    ws.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")

    # ret.pkl — columns report_figure + portfolio_value need
    rng = np.random.default_rng(1)
    pd.DataFrame(
        {
            "account": 1e8 * (1 + np.cumsum(rng.normal(0, 1e-3, len(dates)))),
            "return": rng.normal(3e-4, 1e-2, len(dates)),
            "turnover": rng.uniform(0, 0.2, len(dates)),
            "total_turnover": np.cumsum(rng.uniform(0, 0.2, len(dates))),
            "cost": rng.uniform(0, 1e-4, len(dates)),
            "total_cost": np.cumsum(rng.uniform(0, 1e-4, len(dates))),
            "value": 1e8 * np.ones(len(dates)),
            "cash": 5e6 * np.ones(len(dates)),
            "bench": rng.normal(2e-4, 1e-2, len(dates)),
        },
        index=pd.Index(dates, name="datetime"),
    ).to_pickle(ws / "ret.pkl")

    # pred/label — a signal whose score is (rank_ic-)correlated with next return
    insts = [f"S{i:02d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    n = len(idx)
    score = rng.normal(0, 1, n)
    label = rank_ic * score + np.sqrt(max(1 - rank_ic**2, 0)) * rng.normal(0, 1, n)
    pd.DataFrame({"score": score}, index=idx).to_pickle(ws / "pred.pkl")
    pd.DataFrame({"LABEL0": label}, index=idx).to_pickle(ws / "label.pkl")

    # long_short_r.pkl lives under a sig_analysis/ subdir in real workspaces
    (ws / "sig_analysis").mkdir(exist_ok=True)
    pd.Series(rng.normal(5e-4, 5e-3, len(dates)), index=pd.Index(dates, name="datetime")).to_pickle(
        ws / "sig_analysis" / "long_short_r.pkl"
    )
    return ws


# A qlib config exercising both the jinja (US) and concrete parse paths.
_CONF = """
    provider_uri: "~/.qlib/qlib_data/us_data"
    region: us
market: &market universe
benchmark: &benchmark SPX
                        label:
                            - ["Ref($close, -2)/Ref($close, -1) - 1"]
                - class: qlib.contrib.data.loader.Alpha158DL
            class: TopkDropoutStrategy
            topk: {{ topk | default(30, true) }}
            n_drop: {{ n_drop | default(3, true) }}
            open_cost: {{ open_cost | default(0.0005, true) }}
            close_cost: {{ close_cost | default(0.0005, true) }}
            deal_price: close
        class: LGBModel
"""


def _make_loop(root: Path, ledger_path: Path, loop: int, promote: bool, *, write_run: bool = True,
               write_md: bool = True, parent: int | None = None) -> dict:
    """Append a trial to the ledger and (optionally) drop its run dir + report.md by convention."""
    ws_sel = _make_workspace(root, f"sel{loop}")
    ws_hold = _make_workspace(root, f"hold{loop}")
    (ws_sel / "conf_test.yaml").write_text(_CONF)
    seg_sel = {"train_start": "2015-01-01", "train_end": "2021-12-31",
               "valid_start": "2022-01-01", "valid_end": "2023-12-31",
               "test_start": "2024-01-01", "test_end": "2024-12-31"}
    seg_hold = {"test_start": "2025-01-01", "test_end": "2026-06-01"}
    sel_metrics = {"Rank IC": 0.011, "IC": 0.015, "1day.excess_return_with_cost.information_ratio": 1.2}
    hold_metrics = {"Rank IC": 0.023, "IC": 0.025, "1day.excess_return_with_cost.information_ratio": 1.5,
                    "Long-Short Ann Sharpe": 2.3}

    trial = {
        "loop": loop, "action": "factor", "hypothesis": f"hyp {loop}", "decision": promote,
        "sr_period": 0.08, "sr_ann": 1.27,
        "selection_segment": seg_sel, "selection_metrics": sel_metrics,
        "holdout_segment": seg_hold, "holdout_metrics": hold_metrics,
        "guardrail": {"dsr": 0.77, "gates": {"holdout_ok": True, "dsr_ok": promote}},
        "candidate_workspace": str(ws_sel),
        "timestamp": "2026-07-07T00:00:00+00:00",
    }
    if parent is not None:
        trial["parent_loop"] = parent
        trial["title"] = f"branch of {parent}"

    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"trials": [], "sota_loop": None}
    ledger["trials"].append(trial)
    if promote:
        ledger["sota_loop"] = loop
    ledger.setdefault("baselines", {"bar": 1.83, "bar_set_by": "momentum",
                                    "panel": {"buy_hold": 0.0, "momentum": 1.83}})
    ledger_path.write_text(json.dumps(ledger, indent=2))

    if write_run:
        rundir = R.run_dir_for(ledger_path, loop, trial)
        rundir.mkdir(parents=True, exist_ok=True)
        (rundir / "sel.json").write_text(json.dumps({"workspace": str(ws_sel), "segments": seg_sel,
                                                     "conf": "conf_test.yaml", "settings": {"topk": 25},
                                                     "metrics": {**sel_metrics, "ICIR": 0.12}}))
        (rundir / "hold.json").write_text(json.dumps({"workspace": str(ws_hold), "segments": seg_hold,
                                                      "metrics": {**hold_metrics, "ICIR": 0.18}}))
        (rundir / "decision.json").write_text(json.dumps({
            "decision": promote,
            "dsr": {"value": 0.77, "n_trials": loop + 1, "benchmark_sr_ann": 0.52},
            "gates": {"holdout_ok": True, "dsr_ok": promote, "net_positive": True,
                      "beats_sota_net": True, "beats_baselines": False, "neutral_alpha_ok": True},
            "baselines": {"buy_hold": 0.0, "momentum": 1.83},
            "reasons": ["DSR=0.77 < 0.95", "net IR=+1.5 > 0", "beats baselines: 1.5 <= 1.83"],
        }))
        if write_md:
            (rundir / "report.md").write_text(f"# Loop {loop}\n\nDiscussion for loop {loop}.")
    return trial


# --------------------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------------------
def test_discover_ledgers_and_project_name(tmp_path: Path):
    (tmp_path / "trace.json").write_text('{"trials": [], "sota_loop": null}')
    (tmp_path / "trace_us.json").write_text('{"trials": [], "sota_loop": null}')
    (tmp_path / "notes.txt").write_text("ignore me")
    ledgers = R.discover_ledgers(str(tmp_path))
    assert [p.name for p in ledgers] == ["trace.json", "trace_us.json"]
    assert R.project_name(tmp_path / "trace.json") == "cn"
    assert R.project_name(tmp_path / "trace_us.json") == "us"


def test_run_dir_for_convention_and_override(tmp_path: Path):
    assert R.run_dir_for(tmp_path / "trace_us.json", 3).name == "loop_3"
    assert R.run_dir_for(tmp_path / "trace_us.json", 3).parent.name == "runs_us"
    assert R.run_dir_for(tmp_path / "trace.json", 1).parent.name == "runs"
    override = R.run_dir_for(tmp_path / "trace_us.json", 3, {"run_dir": str(tmp_path / "custom")})
    assert override == tmp_path / "custom"


# --------------------------------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------------------------------
def test_load_report_full_assembly(tmp_path: Path):
    ledger = tmp_path / "trace_us.json"
    _make_loop(tmp_path, ledger, 0, promote=False)
    _make_loop(tmp_path, ledger, 1, promote=False, parent=0)

    rep = R.load_report(ledger, 1)
    assert rep.report_id == "us/1"
    assert rep.verdict == "reject"
    assert rep.parent_loop == 0 and rep.title == "branch of 0"
    # full metric set comes from the run JSON (ICIR present), not the compact trial subset
    assert "ICIR" in rep.selection_metrics
    assert set(rep.gates) == {"holdout_ok", "dsr_ok", "net_positive", "beats_sota_net",
                              "beats_baselines", "neutral_alpha_ok"}
    assert rep.dsr["value"] == 0.77 and rep.dsr["n_trials"] == 2
    assert len(rep.reasons) == 3
    assert rep.baselines["bar_set_by"] == "momentum"
    assert rep.holdout_workspace is not None
    assert rep.report_md and "Discussion for loop 1" in rep.report_md
    assert rep.overfit_gap is not None


def test_verdict_sota(tmp_path: Path):
    ledger = tmp_path / "trace_us.json"
    _make_loop(tmp_path, ledger, 0, promote=True)
    rep = R.load_report(ledger, 0)
    assert rep.is_sota and rep.verdict == "SOTA"


def test_load_report_tolerates_missing_run_json(tmp_path: Path):
    """Only the ledger trial exists (no run dir): falls back to compact metrics + candidate_workspace."""
    ledger = tmp_path / "trace_us.json"
    _make_loop(tmp_path, ledger, 0, promote=False, write_run=False)
    rep = R.load_report(ledger, 0)
    assert rep.report_md is None
    assert rep.selection_metrics.get("Rank IC") == 0.011  # compact set from the trial
    assert rep.selection_workspace is not None  # candidate_workspace fallback
    assert rep.reasons == []


def test_load_all(tmp_path: Path):
    ledger = tmp_path / "trace_us.json"
    _make_loop(tmp_path, ledger, 0, promote=False)
    _make_loop(tmp_path, ledger, 1, promote=True)
    reps = R.load_all(ledger)
    assert [r.loop for r in reps] == [0, 1]
    assert reps[1].is_sota


# --------------------------------------------------------------------------------------------------
# artifact loaders + figures
# --------------------------------------------------------------------------------------------------
def test_artifact_loaders(tmp_path: Path):
    ws = _make_workspace(tmp_path, "probe")
    assert R.equity_frame(str(ws)) is not None
    pred, label = R.signal_frames(str(ws))
    assert list(pred.columns) == ["score"] and list(label.columns) == ["LABEL0"]
    assert R.long_short_series(str(ws)) is not None
    assert R.positions_frame(str(ws)) is None  # Tier-2 not present
    assert R.equity_frame(None) is None


def test_daily_ic_positive_for_correlated_signal(tmp_path: Path):
    ws = _make_workspace(tmp_path, "corr", rank_ic=0.25)
    pred, label = R.signal_frames(str(ws))
    ic = R.daily_ic(pred, label)
    assert not ic.empty
    assert ic["Rank IC"].mean() > 0.05  # the injected correlation shows up


def test_figure_builders_return_figures_and_degrade(tmp_path: Path):
    ws = _make_workspace(tmp_path, "figs")
    eq = R.equity_frame(str(ws))
    pred, label = R.signal_frames(str(ws))
    ls = R.long_short_series(str(ws))
    for fig in (R.cumulative_return_figure(eq), R.drawdown_figure(eq), R.turnover_figure(eq),
                R.qlib_standard_report_figure(eq), R.ic_timeseries_figure(pred, label),
                R.quantile_return_figure(pred, label), R.long_short_cum_figure(ls),
                R.portfolio_value_figure(eq)):
        assert isinstance(fig, go.Figure) and len(fig.data) >= 1
    # every labelled figure carries an axis or bar title (Problem 2: no unlabeled plots)
    ret_fig = R.cumulative_return_figure(eq)
    assert ret_fig.layout.xaxis.title.text and ret_fig.layout.yaxis.title.text and ret_fig.layout.title.text
    # graceful on missing inputs
    assert isinstance(R.ic_timeseries_figure(None, None), go.Figure)
    assert isinstance(R.cumulative_return_figure(None), go.Figure)


def test_parse_conf_handles_jinja_and_concrete():
    assert R._parse_conf(None) == {}
    us = R._parse_conf(_CONF)
    assert us["market"] == "universe" and us["benchmark"] == "SPX" and us["region"] == "us"
    assert us["topk"] == "30" and us["open_cost"] == "0.0005"  # jinja defaults
    assert us["handler"] == "Alpha158" and "LGBModel" in us["model"] and us["strategy"] == "TopkDropout"
    cn = R._parse_conf("region: cn\nmarket: &market csi300\nbenchmark: &benchmark SH000300\n"
                       "topk: 50\nclose_cost: 0.0015\n")
    assert cn["market"] == "csi300" and cn["benchmark"] == "SH000300"
    assert cn["topk"] == "50" and cn["close_cost"] == "0.0015"  # concrete values


def test_backtest_setup_config_segments_and_override(tmp_path: Path):
    ledger = tmp_path / "trace_us.json"
    _make_loop(tmp_path, ledger, 0, promote=False)
    su = R.backtest_setup(R.load_report(ledger, 0))
    cfg = su["config"]
    assert cfg["benchmark"] == "SPX" and cfg["benchmark_name"].startswith("S&P 500")
    assert cfg["n_drop"] == "3"          # from the config's jinja default
    assert cfg["topk"] == 25             # explicit run setting overrides the config default (30)
    assert su["segments"]["train"] and su["segments"]["selection_test"] and su["segments"]["holdout_test"]
    assert su["features"] is None        # no features.json written in this fixture


def test_strategy_explainer_describes_mechanism(tmp_path: Path):
    ledger = tmp_path / "trace_us.json"
    _make_loop(tmp_path, ledger, 0, promote=False)
    secs = R.strategy_explainer(R.load_report(ledger, 0))
    heads = [h for h, _ in secs]
    assert any("Entry" in h for h in heads) and any("Exit" in h for h in heads)
    blob = " ".join(b for _, b in secs)
    assert "top 25" in blob                       # topk override (25) flows into the prose
    assert "sold" in blob and "bought" in blob     # entry/exit described
    assert "next-day forward return" in blob       # label decoded to plain English
    assert secs[-1][1] == "hyp 0"                  # this iteration's hypothesis is included


def test_label_description():
    assert "next-day forward return" in R._label_description("Ref($close, -2)/Ref($close, -1) - 1")
    assert "next-period forward return" in R._label_description(None)
    assert "`$close/Ref($close,5)`" in R._label_description("$close/Ref($close,5)")


def test_segment_returns_excess_identity(tmp_path: Path):
    ws = _make_workspace(tmp_path, "seg")
    seg = R.segment_returns(str(ws))
    assert seg is not None
    # excess is strategy minus benchmark, by construction
    assert abs(seg["excess_net"] - (seg["strategy_net"] - seg["benchmark"])) < 1e-9
    assert isinstance(seg["account_change"], float)
    assert R.segment_returns(None) is None


# --------------------------------------------------------------------------------------------------
# Tier-2 price + entry/exit
# --------------------------------------------------------------------------------------------------
def test_price_with_trades_marks_entry_and_exit(tmp_path: Path):
    dates = list(pd.date_range("2024-01-01", periods=8, freq="B"))
    # AAA held on days 0,1,2 and 5,6 (a gap) inside a universe that spans all 8 days via BBB.
    rows = []
    for i, d in enumerate(dates):
        rows.append({"datetime": d, "instrument": "BBB", "amount": 100})
        if i in (0, 1, 2, 5, 6):
            rows.append({"datetime": d, "instrument": "AAA", "amount": 50})
    positions = pd.DataFrame(rows)
    prices = pd.DataFrame({"datetime": dates * 2,
                           "instrument": ["AAA"] * 8 + ["BBB"] * 8,
                           "close": list(np.linspace(10, 17, 8)) + list(np.linspace(20, 27, 8))})

    entries, exits = R._held_intervals(positions, "AAA")
    assert entries == [dates[0], dates[5]]
    assert exits == [dates[3], dates[7]]

    fig = R.price_with_trades_figure(positions, prices, "AAA")
    names = {t.name for t in fig.data}
    assert {"AAA close", "entry", "exit"} <= names
    assert R.held_tickers(positions) == ["AAA", "BBB"]


def test_price_with_trades_graceful_without_positions():
    fig = R.price_with_trades_figure(None, None, "AAA")
    assert isinstance(fig, go.Figure)
