"""
rdagent-guardrail — the deterministic promote/reject gate (ADR 0002 Phase B; ADR 0001 §7 v0).

This is the scientific crux of the agent-driven loop: it replaces RD-Agent's LLM-judge summarizer
(which lets an LLM eyeball test-segment metrics — textbook selection-on-the-test-set) with a
*deterministic* rule. There is NO LLM here.

A candidate is promoted to SOTA only if ALL gates pass:

  1. holdout_ok      — beats the current SOTA on a LOCKED holdout segment the loop never selects on
                       (or clears an absolute floor if there is no incumbent).
  2. dsr_ok          — its Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) exceeds a threshold.
                       DSR haircuts the Sharpe for (a) the number of trials tried so far and (b) the
                       non-normality (skew/kurtosis) of returns. As trial count grows, the bar rises,
                       so marginal "winners" found by searching get rejected.
  3. net_positive    — its net-of-cost information ratio is positive (reject pure gross winners).
  4. beats_sota_net  — its net-of-cost IR beats the incumbent's by a margin.
  5. beats_baselines — (when a --baselines panel is supplied) its HOLDOUT net-of-cost IR beats the max
                       of a fixed panel of trivial strategies {buy_hold, eqw_1n, momentum, random_p95}
                       (from agent_loop.baselines). Ensures a discovery beats naive strategies, not just
                       the previous champion — a fair, absolute bar. Skipped if no panel is given.

Inputs are the JSON emitted by `agent_loop.run_qlib` (which records the workspace path, so the raw
daily-return series in `<workspace>/ret.pkl` is available for honest T / skew / kurtosis).

Trace JSON schema (shared with the Phase C `trace` tool) — all fields optional except `trials`:
    {"trials": [{"loop": int, "action": str, "decision": bool,
                 "sr_period": float, "holdout_metrics": {metric: value}, ...}],
     "sota_loop": int | null}
N (number of trials for the DSR haircut) = len(trials) + 1 (this candidate).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


# --------------------------------------------------------------------------- stats
def daily_excess_returns(workspace: str | Path, with_cost: bool = True) -> pd.Series:
    """Daily excess return series from a run's ret.pkl (qlib report_normal_1day).

    excess_with_cost = return - bench - cost ;  excess_without_cost = return - bench.
    """
    ret_path = Path(workspace) / "ret.pkl"
    if not ret_path.exists():
        raise FileNotFoundError(f"ret.pkl not found in workspace: {ret_path}")
    df = pd.read_pickle(ret_path)
    exc = df["return"] - df["bench"]
    if with_cost:
        exc = exc - df["cost"]
    return exc.dropna()


def sharpe_stats(returns: pd.Series) -> dict[str, float]:
    """Per-period Sharpe plus the higher moments the DSR standard error needs."""
    r = np.asarray(returns, dtype=float)
    T = int(r.size)
    mean, std = float(r.mean()), float(r.std(ddof=1))
    sr = mean / std if std > 0 else 0.0
    return {
        "T": T,
        "mean": mean,
        "std": std,
        "sr_period": sr,
        "sr_ann": sr * math.sqrt(TRADING_DAYS),
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurt() + 3.0),  # pandas kurt is excess; DSR wants raw (normal=3)
    }


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, T: int, skew: float, kurt: float) -> float:
    """PSR: P(true per-period SR > sr_benchmark), using the Mertens standard error of Sharpe."""
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    denom = max(denom, 1e-12)
    z = (sr - sr_benchmark) * math.sqrt(max(T - 1, 1)) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr_period: float) -> float:
    """Expected maximum per-period Sharpe under the null of `n_trials` noise strategies."""
    if n_trials <= 1 or var_sr_period <= 0:
        return 0.0
    sigma = math.sqrt(var_sr_period)
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(
    stats: dict[str, float], n_trials: int, var_sr_period: float | None
) -> dict[str, Any]:
    """DSR = PSR against the expected-max-Sharpe benchmark implied by multiple testing."""
    sr, T, skew, kurt = stats["sr_period"], stats["T"], stats["skew"], stats["kurt"]
    if var_sr_period is None:
        # No trial-dispersion estimate yet: assume trials are null noise, Var(SR)~1/T.
        var_sr_period = 1.0 / max(T - 1, 1)
        var_source = "default_1/T"
    else:
        var_source = "trace_trial_variance"
    sr0 = expected_max_sharpe(n_trials, var_sr_period)
    dsr = probabilistic_sharpe_ratio(sr, sr0, T, skew, kurt)
    return {
        "value": dsr,
        "n_trials": n_trials,
        "benchmark_sr_period": sr0,
        "benchmark_sr_ann": sr0 * math.sqrt(TRADING_DAYS),
        "var_sr_period": var_sr_period,
        "var_sr_source": var_source,
    }


# --------------------------------------------------------------------------- helpers
def _load(obj: str | dict | None) -> dict | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return json.loads(Path(obj).read_text())


_NET_IR = "1day.excess_return_with_cost.information_ratio"


def _ir_with_cost(res: dict) -> float:
    return float(res["metrics"][_NET_IR])


def _net_ir_of(obj: dict | None) -> float | None:
    """Net-of-cost IR from a run JSON (`metrics`) or a trace trial (`selection_metrics`)."""
    if not isinstance(obj, dict):
        return None
    for key in ("metrics", "selection_metrics"):
        m = obj.get(key)
        if m and _NET_IR in m:
            return float(m[_NET_IR])
    return None


def _sota_from_trace(trace: dict | None) -> dict | None:
    if not trace or not trace.get("trials"):
        return None
    accepted = [t for t in trace["trials"] if t.get("decision")]
    if not accepted:
        return None
    sota_loop = trace.get("sota_loop")
    if sota_loop is not None:
        for t in accepted:
            if t.get("loop") == sota_loop:
                return t
    return accepted[-1]


# --------------------------------------------------------------------------- gate
def evaluate(
    candidate: str,
    holdout: str | None = None,
    trace: str | None = None,
    sota: str | None = None,
    baselines: str | None = None,
    dsr_threshold: float = 0.95,
    cost_margin: float = 0.0,
    holdout_metric: str = "Rank IC",
    min_holdout: float = 0.0,
    out: str | None = None,
) -> dict[str, Any]:
    """Decide whether `candidate` should replace the current SOTA. Returns a decision dict.

    candidate/holdout/sota: paths to run_qlib JSON (selection segment, locked-holdout segment, and
    the incumbent's selection JSON, respectively). trace: path to the trace ledger JSON.
    baselines: path to a baselines.json (from agent_loop.baselines) — the trivial-baseline panel the
    candidate must beat on the holdout (buy_hold / eqw_1n / momentum / random_p95). When supplied, adds
    the `beats_baselines` gate so a discovery must beat naive strategies, not just the champion.
    """
    cand = _load(candidate)
    hold = _load(holdout)
    trace_d = _load(trace)
    # Two SOTA references from different places: the incumbent's *selection* metrics come from a
    # run JSON (--sota), while its *holdout* metrics live on the accepted trace trial.
    sota_trial = _sota_from_trace(trace_d)
    sota_sel = _load(sota) or sota_trial

    stats = sharpe_stats(daily_excess_returns(cand["workspace"], with_cost=True))
    n_trials = (len(trace_d["trials"]) + 1) if (trace_d and trace_d.get("trials")) else 1

    var_sr = None
    if trace_d and trace_d.get("trials"):
        srs = [t["sr_period"] for t in trace_d["trials"] if t.get("sr_period") is not None]
        if len(srs) >= 2:
            var_sr = float(np.var(srs, ddof=1))
    dsr = deflated_sharpe_ratio(stats, n_trials, var_sr)

    reasons: list[str] = []

    # gate: DSR
    dsr_ok = dsr["value"] >= dsr_threshold
    reasons.append(
        f"DSR={dsr['value']:.4f} {'>=' if dsr_ok else '<'} {dsr_threshold} "
        f"(N={n_trials}, benchmark SR_ann={dsr['benchmark_sr_ann']:.3f}, var_src={dsr['var_sr_source']})"
    )

    # gate: net-of-cost positive
    cand_ir_net = _ir_with_cost(cand)
    net_positive = cand_ir_net > 0.0
    reasons.append(f"net IR(with cost)={cand_ir_net:+.4f} {'> 0' if net_positive else '<= 0 (reject gross-only)'}")

    # gate: beats SOTA net-of-cost. Incumbent net IR comes from the --sota run JSON or, failing that,
    # the accepted trace trial's selection_metrics (so the ledger alone can drive the gate).
    sota_ir_net = next((v for v in (_net_ir_of(sota_sel), _net_ir_of(sota_trial)) if v is not None), None)
    have_incumbent = sota_ir_net is not None
    if not have_incumbent:
        sota_ir_net = 0.0
    beats_sota_net = cand_ir_net > sota_ir_net + cost_margin
    reasons.append(
        f"net IR beats SOTA: {cand_ir_net:+.4f} vs {sota_ir_net:+.4f}+{cost_margin} -> {beats_sota_net}"
        + ("" if have_incumbent else " (no incumbent -> bar is 0)")
    )

    # gate: locked holdout
    if hold is None:
        holdout_ok = False
        reasons.append("HOLDOUT MISSING -> cannot promote (a locked-holdout run is required)")
    else:
        cand_h = float(hold["metrics"].get(holdout_metric))
        sota_h = None
        for src in (sota_trial, sota_sel):  # prefer the accepted trial's recorded holdout metrics
            if isinstance(src, dict) and src.get("holdout_metrics"):
                sota_h = src["holdout_metrics"].get(holdout_metric)
                if sota_h is not None:
                    break
        bar = sota_h if sota_h is not None else min_holdout
        holdout_ok = cand_h > bar
        reasons.append(
            f"holdout {holdout_metric}={cand_h:+.4f} {'>' if holdout_ok else '<='} "
            f"{bar:+.4f} ({'SOTA holdout' if sota_h is not None else 'floor'})"
        )

    # gate: beats the trivial-baseline panel on the HOLDOUT (fair-comparison gate). The candidate's
    # out-of-sample net IR must exceed the max of {buy_hold, eqw_1n, momentum, random_p95}.
    base_d = _load(baselines)
    beats_baselines = True  # skipped (not evaluated) when no panel is supplied
    if base_d is not None:
        bar_base = float(base_d["bar"])
        cand_hold_ir = _net_ir_of(hold) if hold else None
        if cand_hold_ir is None:
            beats_baselines = False
            reasons.append("baseline panel supplied but candidate holdout net IR missing -> fail")
        else:
            beats_baselines = cand_hold_ir > bar_base
            reasons.append(
                f"beats baselines (holdout net IR): {cand_hold_ir:+.4f} {'>' if beats_baselines else '<='} "
                f"{bar_base:+.4f} (bar={base_d.get('bar_set_by')}; panel={ {k: round(v,3) for k,v in base_d['baselines'].items()} })"
            )

    decision = bool(holdout_ok and dsr_ok and net_positive and beats_sota_net and beats_baselines)

    result = {
        "decision": decision,
        "candidate_workspace": cand["workspace"],
        "n_trials": n_trials,
        "candidate": {
            "sr_period": stats["sr_period"],
            "sr_ann": stats["sr_ann"],
            "T": stats["T"],
            "skew": stats["skew"],
            "kurt": stats["kurt"],
            "ir_with_cost": cand_ir_net,
        },
        "dsr": dsr,
        "gates": {
            "holdout_ok": holdout_ok,
            "dsr_ok": dsr_ok,
            "net_positive": net_positive,
            "beats_sota_net": beats_sota_net,
            "beats_baselines": beats_baselines,
        },
        "baselines": (base_d.get("baselines") if base_d else None),
        "reasons": reasons,
    }

    print(f"\n==== guardrail: {'PROMOTE' if decision else 'REJECT'} ====")
    for r in reasons:
        print(" -", r)
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2))
        print(f"\n[guardrail] wrote {out}")
    return result


if __name__ == "__main__":
    fire.Fire(evaluate)
