"""
Tests for the trace ledger (ADR 0002 Phase C).

Self-contained: fabricates run JSONs with a ret.pkl (so record() can compute sr_period) in tmp_path.
Verifies append/SOTA semantics and that a recorded ledger feeds the real trial count + SOTA back into
the guardrail. No Docker, no LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_loop import guardrail, trace


def _make_run(root: Path, name: str, sr_ann: float, rank_ic: float = 0.03, T: int = 243) -> str:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    z = rng.standard_normal(T)
    z = (z - z.mean()) / z.std(ddof=1)
    sigma = 0.01
    ret = (sr_ann / np.sqrt(252)) * sigma + sigma * z
    pd.DataFrame(
        {"return": ret, "bench": np.zeros(T), "cost": np.zeros(T)},
        index=pd.date_range("2019-01-01", periods=T, freq="B"),
    ).to_pickle(d / "ret.pkl")
    p = root / f"{name}.json"
    p.write_text(json.dumps({
        "workspace": str(d), "conf": "synthetic", "segments": {"test_start": "2018-01-01"},
        "metrics": {"1day.excess_return_with_cost.information_ratio": sr_ann, "Rank IC": rank_ic, "IC": 0.02},
    }))
    return str(p)


def _decision(promote: bool, dsr: float = 0.98) -> dict:
    return {"decision": promote, "dsr": {"value": dsr},
            "gates": {"holdout_ok": promote, "dsr_ok": True, "net_positive": True, "beats_sota_net": promote}}


def test_init_creates_empty_ledger(tmp_path: Path):
    path = str(tmp_path / "trace.json")
    trace.init(path)
    led = json.loads(Path(path).read_text())
    assert led == {"trials": [], "sota_loop": None}


def test_record_appends_and_sets_sota(tmp_path: Path):
    path = str(tmp_path / "trace.json")
    sel = _make_run(tmp_path, "sel0", sr_ann=1.5, rank_ic=0.030)
    hold = _make_run(tmp_path, "hold0", sr_ann=0.0, rank_ic=0.028)
    trace.record(candidate=sel, holdout=hold, decision=_decision(True), hypothesis="base", action="factor", path=path)
    led = json.loads(Path(path).read_text())
    assert len(led["trials"]) == 1 and led["sota_loop"] == 0
    assert led["trials"][0]["holdout_metrics"]["Rank IC"] == 0.028
    assert led["trials"][0]["sr_period"] != 0.0  # computed from ret.pkl

    # a rejected trial appends but does not move SOTA
    sel2 = _make_run(tmp_path, "sel1", sr_ann=2.0, rank_ic=0.040)
    hold2 = _make_run(tmp_path, "hold1", sr_ann=0.0, rank_ic=0.020)
    trace.record(candidate=sel2, holdout=hold2, decision=_decision(False), hypothesis="cand", path=path)
    led = json.loads(Path(path).read_text())
    assert len(led["trials"]) == 2 and led["sota_loop"] == 0 and led["trials"][1]["decision"] is False


def test_ledger_feeds_guardrail_trial_count_and_sota(tmp_path: Path):
    """After recording 2 trials, the guardrail reading that ledger sees N=3 and the SOTA's holdout."""
    path = str(tmp_path / "trace.json")
    trace.record(candidate=_make_run(tmp_path, "s0", 1.5, 0.030), holdout=_make_run(tmp_path, "h0", 0.0, 0.028),
                 decision=_decision(True), path=path)
    trace.record(candidate=_make_run(tmp_path, "s1", 0.5, 0.010), holdout=_make_run(tmp_path, "h1", 0.0, 0.012),
                 decision=_decision(False), path=path)
    cand = _make_run(tmp_path, "cand", sr_ann=2.6, rank_ic=0.050)
    hold = _make_run(tmp_path, "chold", sr_ann=0.0, rank_ic=0.015)  # below SOTA holdout (0.028)
    res = guardrail.evaluate(candidate=cand, holdout=hold, trace=path)
    assert res["n_trials"] == 3                       # len(trials)+1
    assert res["gates"]["holdout_ok"] is False        # 0.015 <= SOTA-trial holdout 0.028
    assert res["decision"] is False


def test_show_runs(tmp_path: Path, capsys):
    path = str(tmp_path / "trace.json")
    trace.show(path)  # empty
    trace.record(candidate=_make_run(tmp_path, "s0", 1.5, 0.030), holdout=_make_run(tmp_path, "h0", 0.0, 0.028),
                 decision=_decision(True), hypothesis="hy", path=path)
    trace.show(path)
    out = capsys.readouterr().out
    assert "sota_loop" in out and "hy" in out


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i, fn in enumerate([test_init_creates_empty_ledger, test_record_appends_and_sets_sota,
                                test_ledger_feeds_guardrail_trial_count_and_sota]):
            sub = root / f"t{i}"
            sub.mkdir()
            fn(sub)
            print(f"[PASS] {fn.__name__}")
    print("ALL TRACE TESTS PASSED")
