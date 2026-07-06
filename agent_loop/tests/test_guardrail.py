"""
Acceptance tests for the guardrail (ADR 0002 Phase B / ADR 0001 §7).

Self-contained: fabricates a ret.pkl with a controlled daily excess-return series (return-bench-cost)
so we can dial an exact Sharpe and trial count and assert the promote/reject decision. No Docker, no
LLM. Run with `pytest agent_loop/tests/test_guardrail.py` or directly as a script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_loop import guardrail

TRADING_DAYS = 252


def _make_run(root: Path, name: str, sr_ann: float, T: int = 243, holdout_rank_ic: float | None = None) -> str:
    """Workspace with a ret.pkl at an exact per-day Sharpe; returns the run-JSON path."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    z = rng.standard_normal(T)
    z = (z - z.mean()) / z.std(ddof=1)  # standardize -> exact target Sharpe
    sigma = 0.01
    ret = (sr_ann / np.sqrt(TRADING_DAYS)) * sigma + sigma * z
    pd.DataFrame(
        {"return": ret, "bench": np.zeros(T), "cost": np.zeros(T)},
        index=pd.date_range("2019-01-01", periods=T, freq="B"),
    ).to_pickle(d / "ret.pkl")
    metrics = {"1day.excess_return_with_cost.information_ratio": sr_ann}
    if holdout_rank_ic is not None:
        metrics["Rank IC"] = holdout_rank_ic
    p = root / f"{name}.json"
    p.write_text(json.dumps({"workspace": str(d), "conf": "synthetic", "metrics": metrics}))
    return str(p)


def _make_trace(root: Path, n_noise: int, sr_std: float, accepted_holdout_rank_ic: float | None = None) -> str:
    rng = np.random.default_rng(1)
    trials = [
        {"loop": i, "action": "factor", "decision": False, "sr_period": float(rng.normal(0, sr_std))}
        for i in range(n_noise)
    ]
    trace: dict = {"trials": trials, "sota_loop": None}
    if accepted_holdout_rank_ic is not None:
        trials.append(
            {"loop": n_noise, "action": "factor", "decision": True, "sr_period": 0.05,
             "holdout_metrics": {"Rank IC": accepted_holdout_rank_ic}}
        )
        trace["sota_loop"] = n_noise
    p = root / f"trace_{n_noise}.json"
    p.write_text(json.dumps(trace))
    return str(p)


def test_overfit_factor_rejected_by_holdout(tmp_path: Path):
    """High selection Sharpe but poor out-of-sample holdout -> rejected on the holdout gate alone."""
    cand = _make_run(tmp_path, "overfit_cand", sr_ann=2.2)
    hold = _make_run(tmp_path, "overfit_hold", sr_ann=0.0, holdout_rank_ic=0.004)
    trace = _make_trace(tmp_path, n_noise=2, sr_std=0.02, accepted_holdout_rank_ic=0.030)
    res = guardrail.evaluate(candidate=cand, holdout=hold, trace=trace)
    assert res["decision"] is False
    assert res["gates"]["dsr_ok"] and res["gates"]["net_positive"]  # other gates pass
    assert res["gates"]["holdout_ok"] is False  # only holdout fails


def test_clean_winner_promoted_at_low_n(tmp_path: Path):
    cand = _make_run(tmp_path, "win_cand", sr_ann=2.2)
    hold = _make_run(tmp_path, "win_hold", sr_ann=0.0, holdout_rank_ic=0.04)
    res = guardrail.evaluate(candidate=cand, holdout=hold, min_holdout=0.0)
    assert res["decision"] is True


def test_dsr_haircut_rejects_after_many_trials(tmp_path: Path):
    """The SAME Sharpe-2.2 candidate is promoted at N=1 but rejected once 200 trials were tried."""
    cand = _make_run(tmp_path, "cand", sr_ann=2.2)
    hold = _make_run(tmp_path, "hold", sr_ann=0.0, holdout_rank_ic=0.04)
    low = guardrail.evaluate(candidate=cand, holdout=hold, min_holdout=0.0)
    trace = _make_trace(tmp_path, n_noise=200, sr_std=0.064)
    high = guardrail.evaluate(candidate=cand, holdout=hold, trace=trace, min_holdout=0.0)
    assert low["decision"] is True
    assert high["decision"] is False
    assert high["gates"]["dsr_ok"] is False
    assert low["dsr"]["value"] > 0.95 > high["dsr"]["value"]


def test_holdout_compared_against_sota_trial_not_floor(tmp_path: Path):
    """Regression (surfaced by a real Phase-D run): when --sota is a bare run JSON with no holdout
    metrics, the holdout gate must still compare against the SOTA *trial's* recorded holdout metric
    from the trace — not silently fall back to the floor. Candidate improves on selection but is
    below the incumbent on the locked holdout, so it must be rejected on the holdout gate."""
    cand = _make_run(tmp_path, "cand", sr_ann=2.6)                          # strong enough to clear DSR
    hold = _make_run(tmp_path, "hold", sr_ann=0.0, holdout_rank_ic=0.0232)  # below SOTA holdout
    sota = _make_run(tmp_path, "sota", sr_ann=1.0)                          # bare run JSON, no holdout
    trace = _make_trace(tmp_path, n_noise=0, sr_std=0.02, accepted_holdout_rank_ic=0.0244)
    res = guardrail.evaluate(candidate=cand, holdout=hold, sota=sota, trace=trace)
    assert res["gates"]["dsr_ok"] and res["gates"]["net_positive"] and res["gates"]["beats_sota_net"]
    assert res["gates"]["holdout_ok"] is False  # the only failing gate
    assert res["decision"] is False


def test_beats_sota_net_reads_from_trace_trial(tmp_path: Path):
    """With no --sota run JSON, the net-IR gate must read the incumbent's net IR from the accepted
    trace trial's selection_metrics — so the ledger alone drives the gate (Phase E)."""
    cand = _make_run(tmp_path, "cand", sr_ann=1.0)                       # candidate net IR = 1.0
    hold = _make_run(tmp_path, "hold", sr_ann=0.0, holdout_rank_ic=0.05)
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({
        "trials": [{"loop": 0, "action": "factor", "decision": True, "sr_period": 0.05,
                    "selection_metrics": {"1day.excess_return_with_cost.information_ratio": 2.0},
                    "holdout_metrics": {"Rank IC": 0.01}}],
        "sota_loop": 0}))
    res = guardrail.evaluate(candidate=cand, holdout=hold, trace=str(trace_path))
    assert res["gates"]["beats_sota_net"] is False  # 1.0 < SOTA-trial net IR 2.0


def test_missing_holdout_cannot_promote(tmp_path: Path):
    cand = _make_run(tmp_path, "cand", sr_ann=2.2)
    res = guardrail.evaluate(candidate=cand)  # no holdout
    assert res["decision"] is False
    assert res["gates"]["holdout_ok"] is False


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i, fn in enumerate(
            [test_overfit_factor_rejected_by_holdout, test_clean_winner_promoted_at_low_n,
             test_dsr_haircut_rejects_after_many_trials, test_missing_holdout_cannot_promote]
        ):
            sub = root / f"t{i}"
            sub.mkdir()
            fn(sub)
            print(f"[PASS] {fn.__name__}")
    print("ALL GUARDRAIL TESTS PASSED")
