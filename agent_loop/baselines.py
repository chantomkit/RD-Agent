"""
rdagent-baselines — the trivial-baseline panel the candidate must beat (fair-comparison gate).

A discovery is only interesting if it beats naive strategies, not just the previous champion. This
computes {buy_hold, eqw_1n, momentum, random_p95} on the LOCKED holdout, each simulated through the
**same** TopkDropout strategy + cost model as the candidate (only the signal differs), in-process via
qlib's backtest API — faithful and cheap (no Docker). The guardrail then requires the candidate's
holdout net-of-cost IR to exceed the max of the panel.

Panel (net-of-cost information ratio vs the benchmark index, on the holdout):
  - buy_hold   : hold the benchmark index -> excess IR = 0 by construction ("did you beat the market?").
  - eqw_1n     : equal-weight the whole universe (TopkDropout topk=N, n_drop=0, constant signal).
  - momentum   : trailing-return signal through the candidate's exact strategy (topk, n_drop).
  - random_p95 : 95th percentile of IR over many random-signal runs ("better than luck?").

Runs in an env with qlib (the `qlib` conda env):
  python -m agent_loop.baselines --provider us_data --region us --market universe --benchmark SPX \
      --start 2025-01-01 --end 2026-06-01 --out baselines.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _ir(report: pd.DataFrame) -> float:
    exc = report["return"] - report["bench"] - report["cost"]
    exc = exc.dropna()
    s = exc.std()
    return float(exc.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else 0.0


def _run(signal: pd.Series, start: str, end: str, benchmark: str, topk: int, n_drop: int,
         exchange_kwargs: dict) -> float:
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy import TopkDropoutStrategy

    strat = TopkDropoutStrategy(signal=signal, topk=topk, n_drop=n_drop)
    execu = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
    pm, _ = backtest(start, end, strategy=strat, executor=execu, benchmark=benchmark,
                     account=1e8, exchange_kwargs=exchange_kwargs)
    report = pm[list(pm.keys())[0]][0]
    return _ir(report)


def compute(
    provider: str = "us_data",
    region: str = "us",
    market: str = "universe",
    benchmark: str = "SPX",
    start: str = "2025-01-01",
    end: str = "2026-06-01",
    topk: int = 30,
    n_drop: int = 3,
    mom_window: int = 21,
    n_random: int = 30,
    seed: int = 0,
    out: str | None = None,
) -> dict[str, Any]:
    """Compute the baseline panel; returns {baselines, bar, ...}. bar = max the candidate must beat."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=f"~/.qlib/qlib_data/{provider}", region=region)
    exch = dict(deal_price="close", open_cost=0.0005, close_cost=0.0015, min_cost=1, trade_unit=None)

    cal = list(D.calendar(start_time=start, end_time=end, freq="day"))
    insts = D.list_instruments(D.instruments(market=market), start_time=start, end_time=end, as_list=True)
    idx = pd.MultiIndex.from_product([cal, insts], names=["datetime", "instrument"])

    panel: dict[str, float] = {}

    # buy_hold: the benchmark index itself -> zero excess by construction.
    panel["buy_hold"] = 0.0

    # eqw_1n: equal-weight the whole universe (hold all; constant signal, topk = N, no drops).
    const = pd.Series(1.0, index=idx, name="score")
    panel["eqw_1n"] = _run(const, start, end, benchmark, topk=len(insts), n_drop=0, exchange_kwargs=exch)

    # momentum: trailing-return signal through the candidate's exact strategy.
    mom = D.features(D.instruments(market=market),
                     [f"$close/Ref($close,{mom_window})-1"],
                     start_time=start, end_time=end, freq="day").iloc[:, 0].swaplevel().sort_index()
    mom.name = "score"
    panel["momentum"] = _run(mom, start, end, benchmark, topk=topk, n_drop=n_drop, exchange_kwargs=exch)

    # random: many random-signal runs through the candidate's strategy; report mean and p95.
    rng = np.random.default_rng(seed)
    rand_irs = []
    for _ in range(n_random):
        rsig = pd.Series(rng.standard_normal(len(idx)), index=idx, name="score")
        rand_irs.append(_run(rsig, start, end, benchmark, topk=topk, n_drop=n_drop, exchange_kwargs=exch))
    panel["random_mean"] = float(np.mean(rand_irs))
    panel["random_p95"] = float(np.percentile(rand_irs, 95))

    # the bar the candidate must beat = max over the trivial panel (exclude random_mean; p95 is the bar).
    bar_names = ["buy_hold", "eqw_1n", "momentum", "random_p95"]
    bar = max(panel[k] for k in bar_names)
    bar_by = max(bar_names, key=lambda k: panel[k])

    result = {
        "metric": "net_ir_with_cost_holdout",
        "segment": {"start": start, "end": end},
        "config": {"provider": provider, "market": market, "benchmark": benchmark,
                   "topk": topk, "n_drop": n_drop, "mom_window": mom_window, "n_random": n_random,
                   "n_instruments": len(insts)},
        "baselines": panel,
        "bar": bar,
        "bar_set_by": bar_by,
    }
    print("\n==== baseline panel (holdout net IR) ====")
    for k in bar_names + ["random_mean"]:
        print(f"  {k:12s} {panel[k]:+.4f}" + ("   <- BAR" if k == bar_by else ""))
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2))
        print(f"[baselines] wrote {out}")
    return result


if __name__ == "__main__":
    fire.Fire(compute)
