"""
rdagent-trace — the DAG ledger the agent owns (ADR 0002 Phase C).

A plain-JSON, human-inspectable record of every trial (hypothesis + metrics + guardrail verdict). It is
deliberately NOT RD-Agent's `LoopBase` checkpoint machinery (that is coupled to the program-driven
loop). The schema is the same one `agent_loop.guardrail` already reads, so recorded trials feed real `N`
(trial count) and the real SOTA back into the next judgement — closing the loop:

    guardrail.evaluate(candidate, holdout, trace=LEDGER)  ->  decision
    trace.record(candidate, holdout, decision, ..., path=LEDGER)   # append; updates sota_loop

Ledger shape:
    {"trials": [ {loop, action, hypothesis, decision, sr_period, sr_ann,
                  selection_metrics, holdout_metrics, guardrail, ...} ],
     "sota_loop": int | null}

No LLM. Deterministic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fire

from agent_loop.guardrail import daily_excess_returns, sharpe_stats

DEFAULT_PATH = "git_ignore_folder/agent_loop/trace.json"

# Compact metric subsets kept per trial (only those present are stored).
_METRIC_KEYS = [
    "Rank IC",
    "IC",
    "ICIR",
    "Rank ICIR",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]


def _load(obj: str | dict | None) -> dict | None:
    if obj is None or isinstance(obj, dict):
        return obj
    return json.loads(Path(obj).read_text())


def _load_ledger(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"trials": [], "sota_loop": None}
    return json.loads(p.read_text())


def _save_ledger(path: str, ledger: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ledger, indent=2))


def _metric_subset(metrics: dict) -> dict[str, float]:
    return {k: float(metrics[k]) for k in _METRIC_KEYS if k in metrics}


def init(path: str = DEFAULT_PATH) -> str:
    """Create an empty ledger (no-op if one already exists)."""
    if Path(path).exists():
        print(f"[trace] ledger already exists: {path}")
        return path
    _save_ledger(path, {"trials": [], "sota_loop": None})
    print(f"[trace] initialized empty ledger: {path}")
    return path


def record(
    candidate: str,
    holdout: str | None = None,
    decision: str | None = None,
    hypothesis: str = "",
    action: str = "factor",
    path: str = DEFAULT_PATH,
) -> dict[str, Any]:
    """Append a trial to the ledger.

    candidate/holdout: run_qlib JSON for the selection and locked-holdout segments.
    decision: the guardrail's output JSON (for the promote/reject verdict + DSR/gate stats). If omitted,
              the trial is recorded as not-promoted with no guardrail stats.
    """
    cand = _load(candidate)
    hold = _load(holdout)
    verdict = _load(decision)

    ledger = _load_ledger(path)
    loop = len(ledger["trials"])

    stats = sharpe_stats(daily_excess_returns(cand["workspace"], with_cost=True))
    promoted = bool(verdict["decision"]) if verdict is not None else False

    trial: dict[str, Any] = {
        "loop": loop,
        "action": action,
        "hypothesis": hypothesis,
        "decision": promoted,
        "sr_period": stats["sr_period"],
        "sr_ann": stats["sr_ann"],
        "selection_segment": cand.get("segments"),
        "selection_metrics": _metric_subset(cand.get("metrics", {})),
        "holdout_segment": hold.get("segments") if hold else None,
        "holdout_metrics": _metric_subset(hold.get("metrics", {})) if hold else None,
        "guardrail": ({"dsr": verdict["dsr"]["value"], "gates": verdict["gates"]} if verdict else None),
        "candidate_workspace": cand.get("workspace"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ledger["trials"].append(trial)
    if promoted:
        ledger["sota_loop"] = loop
    _save_ledger(path, ledger)

    print(f"[trace] recorded loop {loop} ({action}) -> {'PROMOTED (new SOTA)' if promoted else 'rejected'}; "
          f"{len(ledger['trials'])} trial(s), sota_loop={ledger['sota_loop']}")
    return trial


def show(path: str = DEFAULT_PATH) -> dict[str, Any]:
    """Print the ledger: trial table, SOTA, and selection-vs-holdout overfitting gap."""
    ledger = _load_ledger(path)
    trials = ledger["trials"]
    sota = ledger.get("sota_loop")
    print(f"\n=== trace: {path} ===")
    print(f"trials={len(trials)}  sota_loop={sota}\n")
    if not trials:
        print("(empty)")
        return ledger

    hdr = f"{'loop':>4} {'action':>6} {'verdict':>8} {'sel RankIC':>10} {'hold RankIC':>11} {'gap':>7} {'DSR':>6}  hypothesis"
    print(hdr)
    print("-" * len(hdr))
    gaps = []
    for t in trials:
        sel = (t.get("selection_metrics") or {}).get("Rank IC")
        hol = (t.get("holdout_metrics") or {}).get("Rank IC")
        gap = (sel - hol) if (sel is not None and hol is not None) else None
        if gap is not None:
            gaps.append(gap)
        dsr = (t.get("guardrail") or {}).get("dsr")
        verdict = "SOTA" if t["loop"] == sota else ("promote" if t["decision"] else "reject")
        print(
            f"{t['loop']:>4} {t['action']:>6} {verdict:>8} "
            f"{('%.4f' % sel) if sel is not None else '   n/a':>10} "
            f"{('%.4f' % hol) if hol is not None else '    n/a':>11} "
            f"{('%+.4f' % gap) if gap is not None else '   n/a':>7} "
            f"{('%.3f' % dsr) if dsr is not None else ' n/a':>6}  "
            f"{(t.get('hypothesis') or '')[:48]}"
        )
    if gaps:
        avg = sum(gaps) / len(gaps)
        print(f"\nmean selection-vs-holdout Rank IC gap: {avg:+.4f} "
              f"({'watch for overfitting drift' if avg > 0.01 else 'ok'})")
    return ledger


if __name__ == "__main__":
    fire.Fire({"init": init, "record": record, "show": show})
