---
name: quant-rd-loop
description: Run one iteration of the agent-driven quant R&D loop in this repo — propose a factor/model hypothesis, run selection + LOCKED-holdout qlib backtests, judge with the deterministic guardrail (holdout + Deflated Sharpe + cost), record to the trace ledger, and write a per-loop report the web UI renders. Use when asked to "run the quant loop", "propose and test a factor/alpha", "do a loop iteration", "advance the alpha search", "branch/continue from a report/loop", or to continue the ADR-0002 agent-driven loop. No LLM API key is involved; you (the agent) are the brain.
---

# quant-rd-loop — one iteration of the agent-driven loop

You are the brain of the loop (ADR `docs/proposals/0002-agent-driven-loop.md`). The tools in
`agent_loop/` are deterministic and LLM-free: **you** propose and author code; they execute, judge, and
record. One invocation of this skill = one trial (propose → run → judge → record).

## Fixed environment

```bash
PY=~/anaconda3/envs/rdagent/bin/python
REPO=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO
LEDGER=$REPO/git_ignore_folder/agent_loop/trace.json
```

Needs: Docker up, `local_qlib:latest` image, `~/.qlib/qlib_data/cn_data` extracted. **No `.env` / API key.**
`run_qlib` already bakes in the env fixes (force docker, `MLFLOW_ALLOW_FILE_STORE`, GPU off).

## The scientific invariant — DO NOT VIOLATE

- **Segment policy for a run (freeze at loop 0, never change mid-run):**
  train `2015-01-01..2016-12-31`, valid `2017`, **selection = test 2018**, **LOCKED HOLDOUT = test 2019**.
- You may look at **selection** metrics to reason and iterate. You must **never** look at holdout metrics
  when proposing, and must **never** change the holdout window during a run. The guardrail — not you —
  scores the holdout. This is the whole point (ADR 0001 §7, 0002 §5b).

## Steps

### 0. Read context
```bash
$PY -m agent_loop.trace show --path $LEDGER      # trials, SOTA, selection-vs-holdout gap
```
If the ledger does not exist or is empty, **seed the baseline first** (step S). Otherwise go to step 1.

### S. Seed (only when the ledger is empty) — the PANEL is the initial SOTA; the baseline is VALIDATED
The Alpha20 baseline is **not** auto-promoted (ADR 0002 §5j). The trivial-baseline panel (momentum etc.)
is the initial SOTA floor; the baseline is judged like any candidate and becomes SOTA only if it beats
the panel. Compute the panel (step 2b) first, set it as the floor, then judge + record the baseline:
```bash
N=0; RUN=$REPO/git_ignore_folder/agent_loop/runs/loop_$N; mkdir -p $RUN
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml --train_start 2015-01-01 --train_end 2016-12-31 \
    --valid_start 2017-01-01 --valid_end 2017-12-31 --test_start 2018-01-01 --test_end 2018-12-31 \
    --out $RUN/sel.json                                                   # selection
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml --train_start 2015-01-01 --train_end 2016-12-31 \
    --valid_start 2017-01-01 --valid_end 2017-12-31 --test_start 2019-01-01 --test_end 2019-12-31 \
    --out $RUN/hold.json                                                  # LOCKED holdout
# compute $BASELINES once (see step 2b), then:
$PY -m agent_loop.trace init --path $LEDGER
$PY -m agent_loop.trace set_baselines --baselines $BASELINES --path $LEDGER   # panel = initial SOTA floor
$PY -m agent_loop.guardrail --candidate $RUN/sel.json --holdout $RUN/hold.json \
    --trace $LEDGER --baselines $BASELINES --out $RUN/decision.json
$PY -c "from agent_loop import trace; trace.record(candidate='$RUN/sel.json', holdout='$RUN/hold.json', \
    decision='$RUN/decision.json', hypothesis='Alpha20 base handler', action='factor', path='$LEDGER')"
```

### 1. Propose (this is your job — reason in the open)
- Read `trace show`: what has been tried, where the selection-vs-holdout gaps are widening (overfitting),
  what the SOTA is. Pick `action` = `factor` or `model`.
- **Read the capability substrate `agent_loop/knowledge/` before authoring** (ADR 0002 §5f):
  `operators.md` (the 45 real expression operators + signatures), `fields.md` (the only valid `$fields`
  — heed the absent-field caveat: **this dataset has no `$vwap`**), `handlers.md` (Alpha158/360 base
  catalogs — extend, don't duplicate), `metrics.md`, `models.md`. Author expressions **only** from
  documented operators and fields; never invent a field or operator.
- Write a one-line **hypothesis** and author the change **without touching holdout data**:
  - **Factor via qlib expression (default, simplest):** build a features JSON = Alpha20 + your new
    expression(s). Example:
    ```bash
    N=<next loop id>; RUN=$REPO/git_ignore_folder/agent_loop/runs/loop_$N; mkdir -p $RUN
    $PY -c "import json; from rdagent.utils.qlib import ALPHA20; \
      json.dump({**ALPHA20, 'RET5':'\$close/Ref(\$close,5)-1'}, open('$RUN/features.json','w'))"
    ```
    (Author real, distinct expressions motivated by the hypothesis — not duplicates of Alpha20.)
  - **Factor via arbitrary Python (parquet route):** compute a `combined_factors_df.parquet`
    (index `MultiIndex[datetime, instrument]`, columns `MultiIndex[("feature", name)]`) using the `qlib`
    conda env; **guard the builder with `if __name__ == "__main__":`** (qlib spawn multiprocessing).
    Then pass `--conf conf_combined_factors.yaml --factors <parquet>`.

### 2. Run — selection AND locked holdout (background; each is a Docker backtest ~2–4 min)
```bash
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml --features $RUN/features.json \
    --train_start 2015-01-01 --train_end 2016-12-31 --valid_start 2017-01-01 --valid_end 2017-12-31 \
    --test_start 2018-01-01 --test_end 2018-12-31 --out $RUN/sel.json      # selection
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml --features $RUN/features.json \
    --train_start 2015-01-01 --train_end 2016-12-31 --valid_start 2017-01-01 --valid_end 2017-12-31 \
    --test_start 2019-01-01 --test_end 2019-12-31 --out $RUN/hold.json     # LOCKED holdout
```
Launch these as background jobs and do other work while they run (ADR 0002 §9). Do not block idle.

### 2b. Baseline panel — compute ONCE per (dataset, holdout), then reuse
The candidate must beat trivial strategies, not just the champion. Compute the panel on the LOCKED
holdout (in-process qlib, no Docker; run in the `qlib` env). Cache it and reuse across iterations:
```bash
BASELINES=$REPO/git_ignore_folder/agent_loop/baselines.json
[ -f "$BASELINES" ] || ~/anaconda3/envs/qlib/bin/python -m agent_loop.baselines \
    --provider cn_data --region cn --market csi300 --benchmark SH000300 \
    --start <holdout_start> --end <holdout_end> --out $BASELINES     # (US: --provider us_data --region us --market universe --benchmark SPX)
```

### 3. Judge — the deterministic gate (no LLM)
```bash
$PY -m agent_loop.guardrail --candidate $RUN/sel.json --holdout $RUN/hold.json \
    --trace $LEDGER --baselines $BASELINES --out $RUN/decision.json
```
The trace supplies the real trial count `N`, the SOTA holdout, and the SOTA net IR; the panel supplies
the fair bar. PROMOTE requires ALL of: holdout beats SOTA, Deflated Sharpe ≥ threshold, net-of-cost
IR > 0, beats SOTA net, **and beats the trivial-baseline panel** (buy_hold / eqw_1n / momentum /
random_p95) on the holdout.

### 4. Write the report — the durable artifact, not an in-chat conclusion (ADR 0003)
Do **not** conclude only in the conversation (it evaporates when the session ends). Author a per-loop
**`$RUN/report.md`** — the narrative the web UI renders alongside the figures + gate decision:
- the hypothesis and its rationale (why this factor/model should carry signal);
- the **selection vs. locked-holdout** comparison, called out in words (Rank IC, net IR, L/S Sharpe, and
  the sel−holdout gap — the overfitting fingerprint);
- the guardrail **verdict and which gate decided it** (read `$RUN/decision.json` `reasons`);
- what you'd try next (this seeds a future branch).

Keep it markdown; the numbers/figures come from the JSON + workspace automatically, so focus on judgment.

### 5. Record (with report + lineage)
```bash
$PY -c "from agent_loop import trace; trace.record(candidate='$RUN/sel.json', holdout='$RUN/hold.json', \
    decision='$RUN/decision.json', hypothesis='<your hypothesis>', action='<factor|model>', \
    title='<short UI title>', report='$RUN/report.md', run_dir='$RUN', path='$LEDGER')"
# when branching from a chosen historic loop, also pass parent=<loop id> (see 'Branch' below)
```

### 6. Surface it (point at the report UI; don't re-paste everything)
Tell the user the one-line verdict and the **report id** (`<project>/<loop>`, e.g. `us/2`), and how to
open it. The UI is a read-only viewer over the ledger — no Docker, no `.env`:
```bash
$PY -m agent_loop.report_ui --port 19900     # then open http://127.0.0.1:19900
```
It shows the gate decision, the selection-vs-holdout table, the backtest figures (cumulative return, IC
time series, signal-quantile, long-short spread, and — for US runs — per-ticker price with entry/exit
marks), the discussion you wrote, and a copy-paste **branch** command.

## Branch — fine-tune from a chosen historic loop (ADR 0003)
When the user says "branch from `us/1`" / "continue from loop N" / "refine that report":
1. Read that loop's context: `trace show` and its `runs[_us]/loop_N/report.md` (its hypothesis, params,
   and "what to try next"). **Do not read holdout metrics to pick the new factor** — the invariant holds.
2. Propose the refinement (steps 1–3) seeded from that loop's idea.
3. On record, set **`parent=N`** so the ledger tracks lineage (`trace show` prints "loop M branched from
   loop N"; the UI shows "branched from loop N").

## Rules of thumb
- One hypothesis per iteration; keep factors distinct and motivated.
- Never optimize toward the holdout; if you find yourself reading holdout to pick a factor, stop.
- Budget: each loop = 2 Docker backtests. Cap iterations; re-judging cached runs is free (no Docker).
- If a backtest returns no result, read the run's `docker_execution_*.log` under its workspace.
