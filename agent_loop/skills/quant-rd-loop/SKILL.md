---
name: quant-rd-loop
description: Run one iteration of the agent-driven quant R&D loop in this repo — propose a factor/model hypothesis, run selection + LOCKED-holdout qlib backtests, judge with the deterministic guardrail (holdout + Deflated Sharpe + cost), and record to the trace ledger. Use when asked to "run the quant loop", "propose and test a factor/alpha", "do a loop iteration", "advance the alpha search", or to continue the ADR-0002 agent-driven loop. No LLM API key is involved; you (the agent) are the brain.
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

### S. Seed the incumbent (only when the ledger is empty)
Run the Alpha20 baseline on both segments and record it as loop 0 / SOTA:
```bash
N=0; RUN=$REPO/git_ignore_folder/agent_loop/runs/loop_$N; mkdir -p $RUN
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml --train_start 2015-01-01 --train_end 2016-12-31 \
    --valid_start 2017-01-01 --valid_end 2017-12-31 --test_start 2018-01-01 --test_end 2018-12-31 \
    --out $RUN/sel.json                                                   # selection
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml --train_start 2015-01-01 --train_end 2016-12-31 \
    --valid_start 2017-01-01 --valid_end 2017-12-31 --test_start 2019-01-01 --test_end 2019-12-31 \
    --out $RUN/hold.json                                                  # LOCKED holdout
$PY -m agent_loop.trace init --path $LEDGER
$PY -c "from agent_loop import trace; trace.record(candidate='$RUN/sel.json', holdout='$RUN/hold.json', \
    decision={'decision':True,'dsr':{'value':float('nan')},'gates':{'seed':True}}, \
    hypothesis='Alpha20 base handler (seed SOTA)', action='factor', path='$LEDGER')"
```

### 1. Propose (this is your job — reason in the open)
- Read `trace show`: what has been tried, where the selection-vs-holdout gaps are widening (overfitting),
  what the SOTA is. Pick `action` = `factor` or `model`.
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

### 3. Judge — the deterministic gate (no LLM)
```bash
$PY -m agent_loop.guardrail --candidate $RUN/sel.json --holdout $RUN/hold.json \
    --trace $LEDGER --out $RUN/decision.json
```
The trace supplies the real trial count `N`, the SOTA holdout, and the SOTA net IR. PROMOTE requires
ALL of: holdout beats SOTA, Deflated Sharpe ≥ threshold, net-of-cost IR > 0, and it beats SOTA net.

### 4. Record
```bash
$PY -c "from agent_loop import trace; trace.record(candidate='$RUN/sel.json', holdout='$RUN/hold.json', \
    decision='$RUN/decision.json', hypothesis='<your hypothesis>', action='<factor|model>', path='$LEDGER')"
```

### 5. Report
Summarize to the user: the hypothesis, selection vs holdout metrics, the guardrail verdict and *why*
(which gate decided), and the updated `trace show`. If PROMOTED, it is the new SOTA to beat next loop.

## Rules of thumb
- One hypothesis per iteration; keep factors distinct and motivated.
- Never optimize toward the holdout; if you find yourself reading holdout to pick a factor, stop.
- Budget: each loop = 2 Docker backtests. Cap iterations; re-judging cached runs is free (no Docker).
- If a backtest returns no result, read the run's `docker_execution_*.log` under its workspace.
