# ADR 0002 — The agent *is* the loop: RD-Agent machinery as deterministic tools

- **Status:** Accepted
- **Date:** 2026-07-06
- **Deciders:** repo owner
- **Relationship to 0001:** **Amends [ADR 0001](0001-autonomous-quant-rd-loop.md).** Keeps 0001's
  vision (autonomous qlib alpha search with honest evaluation), its guardrail design (§7), and its
  file map. **Supersedes 0001's execution-mode decision (§6):** we do *not* run RD-Agent's autonomous
  program and swap three LLM components. Instead, **this Claude Code session's agent drives the loop
  directly**, calling RD-Agent's deterministic machinery as tools.

> Like 0001, this is both a decision record and a cold-start handoff. A fresh session should be able
> to read this and continue. Paths are relative to the repo root.

---

## 1. TL;DR decision

RD-Agent's loop routes *all* intelligence through one choke point —
`APIBackend().build_messages_and_create_chat_completion(...)` — i.e. it phones a *second, external*
LLM for every propose / code / judge step. That is what needs `.env` + `ANTHROPIC_API_KEY`, and it is
why `python rdagent/app/qlib_rd_loop/quant.py` fails with no credentials.

**We already have an LLM with tools: this session's agent.** So instead of paying an API to run a
worse-integrated copy of the agent, **the agent becomes the brain** (propose hypotheses, write
factor/model code, judge results) and RD-Agent's **LLM-free machinery** — Dockerized qlib execution,
the trace ledger, and a statistical guardrail — is exposed as **deterministic tools the agent calls**.

Consequences: **no second API key**, every reasoning step is visible in the agent transcript, and two
of 0001's open problems dissolve (see §6). The cost: the loop is **attended and low-throughput**
(a handful of honest iterations per session) rather than **unattended at scale** (hundreds of loops
overnight). We chose transparency + honest evaluation over unattended throughput. If unattended scale
becomes the goal, 0001's program-driven path is the fallback (point LiteLLM at Claude via `.env`).

---

## 2. For the next session — start here

**What is true right now (verified this session):**
- **Slice 1 is proven.** A cold session drove a Dockerized qlib backtest end-to-end and got real
  metrics back, with **no `.env` and no LLM call in the path**. See §4.
- The tools exist: **`agent_loop/run_qlib.py`** (execute), **`agent_loop/guardrail.py`** (the
  deterministic promote/reject gate), and **`agent_loop/trace.py`** (the DAG ledger). All LLM-free.
- **Phases A–F are done.** The guardrail rejects a deliberately overfit factor and applies a real
  Deflated-Sharpe haircut; the trace ledger closes the loop (feeds real `N` + SOTA back to the
  guardrail); the `quant-rd-loop` skill wires one full iteration; and `agent_loop/knowledge/` is a
  qlib capability substrate extracted from the installed qlib. Tests: `agent_loop/tests/`
  (13 passing: 6 guardrail + 4 trace + 3 knowledge).
- `agent_loop/` is a **new top-level package kept outside `rdagent/`** so upstream merges stay cheap
  (0001 §10). Do not put loop-driving logic inside `rdagent/`.

**Environment facts (this machine, verified):**
- Host env: conda env **`rdagent`** (Python 3.10.20), `rdagent` editable-installed
  (`pip install -e .`). Run tools with `~/anaconda3/envs/rdagent/bin/python`. (The `base` env is
  Python 3.11 and is missing deps — do not use it.)
- qlib runs in Docker image **`local_qlib:latest`** (already built). Docker daemon is up.
- qlib data: **`~/.qlib/qlib_data/cn_data`** is extracted (features/instruments/calendars present).
  The container mounts `~/.qlib/` → `/root/.qlib/`.
- **Two env fixes are required and are baked into `run_qlib.py`:**
  1. Force **`env_type=docker`**. The default is `conda` (`MODEL_CoSTEER_ENV_TYPE`), which would
     auto-create a heavy `rdagent4qlib` conda env (qlib+torch). The Docker image is already provisioned.
  2. Pass **`MLFLOW_ALLOW_FILE_STORE=true`** into the container — its newer mlflow refuses the
     file-store tracking backend otherwise (hard-fails the run with no result file).
- GPU: this is a Mac; `QlibDockerConf.enable_gpu` defaults `True` but auto-disables when unavailable.
  `run_qlib.py` sets `QLIB_DOCKER_ENABLE_GPU=False` to skip the probe.

**Next deliverables (in order):**
1. ~~`guardrail` tool~~ — **DONE** (`agent_loop/guardrail.py`). See §5b.
2. ~~`trace` tool~~ — **DONE** (`agent_loop/trace.py`). See §5d.
3. ~~factor injection path~~ — **DONE** (both the `--features` expression route and the `--factors`
   parquet route). See §5c.
4. ~~Claude Code skill `quant-rd-loop`~~ — **DONE** (`agent_loop/skills/quant-rd-loop/SKILL.md`). See §5e.

**Now open:** run real loop iterations at scale via the skill (accumulate a trace, watch DSR/overfitting
drift), and the leakage-safe evaluation upgrade (0001 §7 v1: purged/embargoed CV, embargo ≥ 2 days).

---

## 3. Architecture: who does what

| R&D stage | Brain | Hands (deterministic tool) |
|---|---|---|
| Propose / Formalize | **agent** (reads scenario + knowledge files + trace SOTA, writes hypothesis) | — |
| Implement | **agent** (writes factor expr / `model.py`) | — |
| Execute | — | **`agent_loop/run_qlib.py`** → `QlibFBWorkspace.execute()` → Docker qlib |
| Evaluate (decision) | — | **`guardrail`** (holdout + Deflated Sharpe + cost margin) — *not an LLM* |
| Evaluate (qualitative) | **agent** (observations, next-hypothesis) | — |
| Memorize | — | **`trace`** (JSON/SQLite DAG ledger) |

The agent is `HypothesisGen` + coder + the *qualitative* half of feedback. The promote/reject decision
is deterministic Python — which is exactly what 0001 Gap 1 demanded (stop letting an LLM eyeball
test-segment metrics). Here that is **structural**, not a swapped module.

---

## 4. Verified spine (slice 1)

**The whole execution path is LLM-free.** The only file under `rdagent/scenarios/qlib/developer/`
that imports `APIBackend` is `feedback.py` — the component 0001 replaces. Both runners
(`factor_runner.py`, `model_runner.py`) and `utils/env.py` are LLM-free. Every backtest reduces to:

```python
# rdagent/scenarios/qlib/experiment/workspace.py — QlibFBWorkspace.execute()
result, stdout = exp.experiment_workspace.execute(qlib_config_name="conf_*.yaml", run_env=env_to_use)
exp.result = result   # pandas Series, indexed by metric name
```

`run_env` carries `train_start/end, valid_start/end, test_start/end` and is passed as **container env
vars**, which qlib's `qrun` substitutes into the YAML template's Jinja fields. **This is the holdout
seam:** the guardrail controls which date segments a backtest is allowed to see, with no core edits.

**Proof run** (`conf_baseline.yaml`, Alpha20, train 2015–16 / valid 2017 / test 2018), metrics returned:

```
IC 0.0277   Rank IC 0.0283   ICIR 0.254   Rank ICIR 0.250
1day.excess_return_with_cost.annualized_return -0.0195
1day.excess_return_with_cost.max_drawdown      -0.0797
1day.excess_return_with_cost.information_ratio -0.245
```

(Weak numbers — it's a deliberately tiny baseline; the point is the spine.) The Series contains
**exactly** the `IMPORTANT_METRICS` from `feedback.py` (`IC`, `...with_cost.annualized_return`,
`...with_cost.max_drawdown`) plus Rank IC/ICIR/information_ratio — the guardrail's inputs.

---

## 5. The `run_qlib` tool (built)

`agent_loop/run_qlib.py` — `python -m agent_loop.run_qlib [flags]`. Key flags: `--conf` (template
yaml), `--template factor|model`, `--factors <parquet>` (inject `combined_factors_df.parquet`),
`--features <json>` (default Alpha20), the six date-segment flags, `--env docker|conda`, `--gpu`,
`--timeout`, `--out <json>`. Returns/dumps `{workspace, conf, segments, metrics}`. It mutates
`MODEL_COSTEER_SETTINGS.env_type` at runtime and injects `MLFLOW_ALLOW_FILE_STORE`/GPU env — a caller
never needs a `.env`.

---

## 5b. The `guardrail` tool (built)

`agent_loop/guardrail.py` — `python -m agent_loop.guardrail --candidate <sel.json> --holdout <hold.json>
[--trace <trace.json>] [--sota <sel.json>]`. Deterministic promote/reject; **no LLM**. Promotes only if
**all** gates pass:

1. **holdout_ok** — beats the current SOTA on a locked holdout segment (or clears a floor if no incumbent).
2. **dsr_ok** — Deflated Sharpe Ratio (Bailey & López de Prado 2014) ≥ threshold (default 0.95). DSR
   reads the raw daily excess-return-with-cost series from `<workspace>/ret.pkl` for honest T / skew /
   kurtosis, then haircuts the Sharpe by the trial count `N = len(trace.trials)+1` and the trial-Sharpe
   dispersion. As `N` grows, the bar rises.
3. **net_positive** — net-of-cost information ratio > 0 (reject pure gross winners).
4. **beats_sota_net** — net-of-cost IR beats the incumbent by `--cost_margin`.

**Acceptance results** (`agent_loop/tests/test_guardrail.py`, 4 passing): an overfit factor (strong
selection Sharpe, poor holdout Rank IC) is rejected on the holdout gate; and a Sharpe-2.2 candidate that
is promoted at `N=1` (DSR 0.98) is rejected after 200 trials (DSR 0.35) — the multiple-testing haircut
biting exactly as designed. The trace JSON schema the guardrail reads is the contract for the Phase C
`trace` tool.

## 5c. Phase D — the full arc on a real factor (done)

An agent-authored factor set (5-day return / 5d-20d volume ratio / 10-day amplitude) was added to the
Alpha20 base and driven end-to-end via the expression route (`run_qlib --features`), on a **selection**
segment (2018) and a **locked holdout** (2019), then judged. Result:

| run | Rank IC | net IR |
|---|---|---|
| baseline  selection (2018) | 0.0283 | −0.245 |
| baseline  holdout (2019)   | 0.0244 | +0.286 |
| candidate selection (2018) | **0.0387** | −0.563 |
| candidate holdout (2019)   | 0.0232 | +1.451 |

**Decision: REJECT — and it is exactly the failure mode the loop exists to catch.** The factor *raised*
Rank IC on the selection segment (0.028 → 0.039), the classic in-sample lure a metric-eyeballing judge
would promote on; but on the locked 2019 holdout it did **not** generalize (0.0244 → 0.0232, slightly
worse than baseline). The guardrail rejected it.

**Dogfooding also caught a real guardrail bug.** The holdout gate was comparing the candidate against a
floor of 0 instead of the incumbent's holdout metric, because the SOTA's holdout metric lives on the
trace *trial* (`holdout_metrics`), while the code only looked for it on the `--sota` run JSON. Fixed to
prefer the trace trial; locked with a regression test (`test_holdout_compared_against_sota_trial_not_floor`).
The re-judgement ran in ~1 s on the cached run JSONs — **no Docker** — demonstrating the file-based
resumability from §9 (a candidate's backtests are computed once; re-scoring under a fixed gate is free).

**Parquet route — also validated.** The `run_qlib --factors <combined_factors_df.parquet>` path (for
factors that need arbitrary Python, not just a qlib expression) was confirmed end-to-end with a
qlib-computed factor (a `[datetime, instrument] × [("feature", name)]` parquet), returning real metrics
(IC 0.033, Rank IC 0.037) with no errors. Gotcha for the host-side factor builder: guard it with
`if __name__ == "__main__":` (qlib's `spawn` multiprocessing — §9).

## 5d. The `trace` tool (built)

`agent_loop/trace.py` — `python -m agent_loop.trace {init|record|show}`. A plain-JSON DAG ledger the
agent owns; **not** RD-Agent's `LoopBase` checkpoint machinery (that is coupled to the program loop).
Its schema is the contract `guardrail.py` already reads, so recorded trials feed the real trial count
`N` and the real SOTA back into the next judgement — **closing the loop** without synthetic traces:

```
guardrail.evaluate(candidate, holdout, trace=LEDGER)  ->  decision
trace.record(candidate, holdout, decision, hypothesis=..., path=LEDGER)   # append; updates sota_loop
```

`record` computes each trial's per-period Sharpe from its `ret.pkl`, stores compact selection/holdout
metric subsets + the guardrail verdict, and promotes `sota_loop` on acceptance. `show` prints the trial
table and the **selection-vs-holdout Rank IC gap** — the overfitting-drift signal.

**Real closed-loop demo** (Phase-D run JSONs, no Docker): seeding the Alpha20 baseline as SOTA (loop 0)
then judging the candidate produced this ledger —

```
loop action  verdict sel RankIC hold RankIC     gap    DSR  hypothesis
   0 factor     SOTA     0.0283      0.0244 +0.0039    nan  Alpha20 base handler (seed SOTA)
   1 factor   reject     0.0387      0.0232 +0.0155  0.139  add RET5 + VOLR + AMP10 to Alpha20
```

— the guardrail read the ledger for `N=2` and the SOTA holdout (0.0244), and the candidate's inflated
selection-vs-holdout gap (+0.0155 vs +0.0039) is the visible overfitting fingerprint. Tests:
`agent_loop/tests/test_trace.py` (4 passing), incl. a ledger→guardrail round-trip.

## 5e. The `quant-rd-loop` skill (built)

`agent_loop/skills/quant-rd-loop/SKILL.md` — a Claude Code skill = **the one-iteration recipe** that
makes the agent the brain and the three tools the hands: read `trace show` for context → propose a
hypothesis and author a factor (qlib expression via `--features`, or arbitrary-Python via `--factors`)
**without looking at the holdout** → `run_qlib` on selection + the locked holdout → `guardrail` (reading
the ledger for `N`/SOTA) → `trace record` → report. It encodes the fixed segment policy (selection 2018,
locked holdout 2019) and the scientific invariant (never optimize toward, or change, the holdout).

Repo layout note: the tracked source of truth lives under `agent_loop/skills/` because this repo
gitignores `.claude/`. To activate it in a session, install a copy:

```bash
mkdir -p .claude/skills/quant-rd-loop && cp agent_loop/skills/quant-rd-loop/SKILL.md .claude/skills/quant-rd-loop/
```

The skill's judge step calls `guardrail --candidate … --holdout … --trace $LEDGER` with **no** `--sota`:
the ledger alone supplies `N`, the SOTA holdout, and the SOTA net IR (verified end-to-end via CLI).

## 5f. The qlib knowledge substrate (built — Phase F)

`agent_loop/knowledge/` — a curated, retrievable corpus the agent reads **directly** when proposing
(no embeddings/RAG; that dissolved in §6). It is **extracted from the installed qlib**, not hand-written
— otherwise it would hallucinate the very corpus meant to stop hallucination. `knowledge/build.py`
(run in the `qlib` env) regenerates it:

| file | contents | source |
|---|---|---|
| `operators.md` | 45 expression operators + signatures, by category | extracted from `qlib.data.ops` |
| `fields.md` | the 7 real `$` fields + label convention + **absent-field caveat** | extracted from the dataset |
| `handlers.md` | Alpha158 (158) + Alpha360 (360) catalogs | extracted from `qlib.contrib.data.loader` |
| `metrics.md` / `models.md` | metric glossary + model cards | curated from templates/run output |

**The extractor earns its keep immediately.** Its self-consistency check (every operator/field the
handlers use must be documented) surfaced that the stock Alpha158/360 handlers reference **`$vwap`,
which this cn_data dataset does not contain** — features using it silently become NaN. That is now a
prominent "DO NOT USE" caveat in `fields.md`, not a landmine. Operators are hard-asserted grounded (an
undocumented operator = extractor bug); absent fields are documented as caveats.

The `quant-rd-loop` skill's propose step now instructs the agent to read this substrate first and author
expressions only from documented operators/fields. Tests (`agent_loop/tests/test_knowledge.py`): the
substrate is present and the real base features (`ALPHA20`) are fully grounded in it.

## 5g. Running on recent data / other markets (US path)

The public qlib bundles (`cn_data`, prebuilt `us_data`) are **frozen ~2020**, so "use 2024–2026 data"
required provisioning fresh data. Built and verified end-to-end:

- **`agent_loop/data/build_us.py` + `data/README.md`** — fetch adjusted daily OHLCV from Yahoo
  (yfinance) for a fixed large-cap universe (+ `^GSPC`→`SPX` benchmark) → per-symbol CSVs → qlib binary
  via qlib's `dump_bin.py`. Result: `~/.qlib/qlib_data/us_data`, 102 stocks, calendar to **2026-07-02**.
- **`agent_loop/qlib_templates/us_template/`** — US conf templates (`provider_uri=us_data`, `region=us`,
  `market=universe`, `benchmark=SPX`, no CN price-limit, `topk=30/n_drop=3`). Label + Alpha158 unchanged.
- **`run_qlib --template_dir <dir>`** — new flag to run against a custom template (default cn behavior
  unchanged). The Docker env already mounts `~/.qlib`, so `us_data` is visible with no mount change.

**Verified loop run** — selection **2024**, locked holdout **2025–2026**:

```
loop verdict sel RankIC hold RankIC    DSR   net IR(cost)
   0 SOTA       -0.0023      0.0077    nan   -0.64   (US Alpha20 baseline)
   1 reject     -0.0001      0.0071   0.016  -1.61   (higher-moment/lottery/illiquidity)
```

**Honest finding:** on a small, efficient US large-cap universe, the CN-tuned Alpha features carry
**~no cross-sectional signal** (Rank IC ≈ 0 everywhere), and the candidate's extra factors only added
turnover (net IR −1.6) — REJECTED on all four gates. The loop refused to promote noise; the negative
gap (selection < holdout) confirms there was no in-sample edge to overfit. This points at real next
steps (broader universe, US-appropriate features, lower turnover), which is exactly what the loop is for.

**Caveats (see `data/README.md`):** survivorship bias (fixed current universe), adjusted prices
(`factor=1`), end the holdout a few weeks before the data edge (qlib peeks one day ahead to execute the
final rebalance), and the knowledge substrate is still cn-extracted (regenerate `build.py` against
`us_data` for field-sensitive US factors).

## 5h. Fair comparison — the baseline-panel gate

**Problem (raised in review):** the SOTA gate is a dynamic champion (updated on each promotion), but the
*seed* champion was hand-promoted and a candidate only had to beat the previous champion + the index
zero-line — never a panel of trivial strategies. A discovery could "beat SOTA" while being worse than
a one-line rule.

**Fix:** `agent_loop/baselines.py` computes a fixed panel on the LOCKED holdout, each simulated through
the **same** TopkDropout strategy + cost model as the candidate (only the signal differs), **in-process
via qlib's backtest API — no Docker, ~1 s each**:

- `buy_hold` — hold the benchmark index (excess IR = 0 by construction: "beat the market?").
- `eqw_1n` — equal-weight the whole universe (topk=N, n_drop=0, constant signal).
- `momentum` — trailing-return signal through the candidate's exact strategy.
- `random_p95` — 95th percentile of IR over many random-signal runs ("better than luck?").

The guardrail gains a 5th gate, `beats_baselines`: the candidate's **holdout net-of-cost IR** must
exceed `max(panel)`. Compute the panel once per (dataset, holdout) and reuse.

**This immediately exposed the US result as worse than trivial.** On the 2025–2026 holdout the panel was:

```
buy_hold 0.00   eqw_1n -2.88   momentum +0.785   random_p95 -0.12   =>  BAR = +0.785 (momentum)
```

The candidate's holdout net IR was **−0.53** — i.e. the ML strategy (and the Alpha20 "SOTA" at −0.64)
**lost to a 21-day momentum rule.** Re-judged with the panel, `beats_baselines=false` — the loop now
rejects for the *right, fair* reason. Tests: `test_beats_baselines_gate`, `test_no_baselines_skips_gate`
(suite 15). Follow-ups: seed validation — **done, see §5j**; and a market-neutral (long-short) evaluation
to separate alpha from beta (not yet done).

## 5i. CN→US infrastructure fixes (system-level, not strategy tuning)

Several defaults were CN-baggage that quietly crippled US results. Fixing them at the infra level helps
**every** future US loop:

- **Universe breadth** — `build_us.py --sp500` (Wikipedia list via a browser UA + requests) rebuilds
  `us_data` with the **full S&P500** (500 names + SPX) instead of a 102-name fallback. More names to rank
  = more cross-sectional signal.
- **Cost model** — the cn template's `close_cost=0.0015` encodes CN's 0.1% **sell stamp duty**, absent in
  US. US template now uses **symmetric ~5bps/side** (`open_cost=close_cost=0.0005`, `min_cost=0`).
- **Configurable turnover** — `topk`, `n_drop`, `hold_thresh`, `open_cost`, `close_cost` are now Jinja
  params injected by `run_qlib` (not hard-coded), so the loop can *explore* turnover (e.g. raise
  `hold_thresh` to cut it) instead of me picking a value.
- **US-aware substrate** — `knowledge/build.py --providers cn_data,us_data` regenerates a **per-dataset**
  `fields.md` (US has no `$change`; neither has `$vwap`), so proposals reference fields that exist.

**Impact — the same Alpha20 baseline, OOS on 2024 (trained 2015–21):**

| | 102 names + CN cost | S&P500 + US cost |
|---|---|---|
| gross IR | +0.26 | **+0.97** |
| **net IR** | **−0.64** | **+0.75** |
| net annual return | −3.4% | **+8.5%** |
| Rank IC | −0.002 | **+0.009** |

The earlier "worse than trivial" US result was **mostly the CN mismatch, not absence of signal**: with the
right universe + cost model the baseline reaches **net IR +0.75 OOS — at the momentum bar (0.785)**. This
is the highest-leverage change in the session and validates fixing infrastructure before tuning strategy.
(Next: re-run the full loop + recompute the baseline panel on the 500-name universe.)

## 5j. Seed validation — the baseline panel is the initial SOTA

**Problem (§5h roadmap item):** the loop *hand-seeded* the Alpha20 baseline as SOTA with a forced
`decision=True` — it never passed a gate. So "SOTA" could be an arbitrary strategy that doesn't even beat
momentum, and `trace show` misleadingly labelled it the champion.

**Fix:** the **trivial-baseline panel is the initial SOTA floor**; the baseline is *validated*, not seeded.

- `trace.set_baselines(panel, ledger)` records the panel as the ledger's `baselines` floor (the bar to
  beat). `trace show` prints it and states whether any trial has beaten it.
- The seed step now runs the Alpha20 baseline through the guardrail (against the panel) like any
  candidate and records the **real** decision. It becomes SOTA only if it beats the panel; otherwise
  `sota_loop` stays `null` and the target remains the panel.

**On the real US S&P500 ledger** this cleans the story up exactly as intended:

```
loop verdict sel RankIC hold RankIC hold netIR   DSR  hypothesis
   0 reject     0.0085     0.0096     +1.446   0.779  US S&P500 Alpha20 baseline
   1 reject     0.0103     0.0233     +1.534   0.769  US: add MOM12_1 + VOL60 to Alpha20
SOTA floor (trivial baselines): +1.8257 (momentum) — beat this to become SOTA
champion: none — nothing beats the floor yet; the target is the floor
```

The Alpha20 baseline (holdout net IR 1.45) is no longer a fake champion — it's rejected like everything
else because it doesn't beat momentum (1.83). Every future verdict is now measured against an honest,
non-arbitrary bar. Tests: `test_seed_validation_panel_is_initial_sota`; suite 16.

## 6. What dissolves vs 0001

- **Embeddings / RAG (0001 §8, §10 open item): gone.** RAG existed to feed a *remote* LLM past
  context. The agent retrieves by reading the knowledge-substrate files directly. Anthropic-has-no-
  embeddings is a non-problem here.
- **LLM-judge eval gap (0001 §7, Gap 1): structural, not a swap.** The decision is deterministic Python
  the agent calls; there is no LLM in the promote/reject path by construction.
- **Still true from 0001:** Docker dependency stays (qlib runs in a container — but that is local
  execution, not an API call). The guardrail's statistical design (holdout + Deflated Sharpe + cost
  margin, purged CV later) is unchanged — see 0001 §7.

---

## 7. What we give up (and the fallback)

Program-driven RD-Agent buys **unattended overnight autonomy**: parallel loops, bandit factor-vs-model
selection, auto-retry evolving coder, hundreds of iterations. Agent-driven is **attended and
low-throughput** by nature. If throughput/scale becomes the goal, the fallback is 0001's path: keep
`quant.py`, point `LiteLLMAPIBackend` at Claude via `.env`, and swap the summarizer for the guardrail
module. The two paths **share** the guardrail and the knowledge substrate, so that work is not wasted.

---

## 8. Phased plan (supersedes 0001 §9 Phases 0–2)

- **Phase A — spine (DONE).** `run_qlib` tool; slice-1 baseline backtest returns real metrics, no key.
- **Phase B — guardrail tool (DONE).** Deterministic holdout + Deflated Sharpe + cost margin (0001 §7
  v0), consuming `run_qlib` metric JSON. *Gate met: overfit factor rejected; DSR haircut blocks a
  marginal winner as trial count grows.* See §5b.
- **Phase C — trace tool (DONE).** Plain-JSON DAG ledger owned by `agent_loop`; closes the loop by
  feeding real `N` + SOTA back to the guardrail. *Gate met: ledger→guardrail round-trip; file-based
  state means a session resumes from the ledger.* See §5d.
- **Phase D — factor path (DONE, expression route).** Agent-authored factor run through the full arc
  (propose → qlib expressions → run on selection + locked holdout → guardrail). See §5c.
- **Phase E — skill (DONE).** `quant-rd-loop` Claude Code skill = the one-iteration recipe over A–D
  (propose → run selection+holdout → guardrail(trace) → record → report). See §5e.
- **Phase F — knowledge substrate (DONE).** qlib operator/field/handler/model-card corpus **extracted
  from the installed qlib** (`agent_loop/knowledge/`), read directly by the agent when proposing; the
  extractor's self-consistency check surfaced the `$vwap`-absent caveat. See §5f.
- **Next: leakage-safe evaluation (0001 §7 v1).** Purged/embargoed CV (embargo ≥ 2 days for the 2-day
  label) in the qlib templates + guardrail — the remaining scientific upgrade.

---

## 9. Operational model — long-running jobs without stalling the session

**The core tension of agent-driven execution:** every loop iteration pairs *cheap* agent reasoning
(propose / code / judge — seconds) with an *expensive* deterministic job (a Dockerized qlib backtest is
2–4 min today on a short segment; a full universe, GPU model training, or a purged-CV sweep will be tens
of minutes to hours). If the session blocks on each job, the agent — the scarce resource — sits idle.
This is a first-class design constraint, not an afterthought, and it shapes how the loop is run.

**How the loop stays active while a job runs:**

- **Background + notify, never block.** Launch backtests as background jobs (`run_in_background`); the
  harness re-invokes the agent when the job exits. The agent does *useful* filler work in the gap:
  propose the next hypothesis, write/refine a tool, update this ADR, or do git/PR housekeeping. (This
  very document was extended, and the branch/PR prepared, while a 4-backtest Phase-D run was executing.)
- **Parallel fan-out.** Independent backtests can run as concurrent background jobs (e.g. baseline vs
  candidate × selection vs holdout — 4 at once). Caveat: qlib's Docker env mounts a shared cache
  (`/tmp/full` → `workspace_cache`); heavy parallelism can race on it, so cap concurrency or isolate
  caches per job.
- **File-based state = interruptible/resumable.** The loop's state is *on disk* (run JSONs, the trace
  ledger), not in a live Python process. A session can end mid-flight and a fresh one resumes from the
  ledger — a structural advantage over the program-driven loop, whose state lives in `LoopBase` memory
  and needs its checkpoint machinery. Design tools to be **idempotent and re-entrant**: write results to
  a stable path, and treat "job already produced its JSON" as success.
- **Polling only when the harness can't notify.** For state the harness cannot observe (a CI run, a
  remote queue), poll with a `Monitor` until-loop. Do **not** hand-roll `sleep` in a foreground shell —
  the harness blocks long foreground sleeps by design.
- **Self-paced loops.** For an unattended cadence, schedule the next iteration with a wake-up timer.
  Match the delay to what you're waiting on and to the prompt-cache window (~5 min): sub-5-min polls
  keep the cache warm for a job about to finish; for long jobs, sleep long (20–60 min) and let the job's
  own completion event be the primary wake signal, with the timer as a fallback.

**Practical gotchas discovered during bring-up (all now handled in `agent_loop/`):**

- Foreground `sleep` is blocked — use background jobs or a `Monitor` until-loop to wait on a condition.
- On macOS, qlib/`D.features` uses the `spawn` multiprocessing start method; any host-side factor-
  computation script **must** guard its body with `if __name__ == "__main__":` or it crashes with
  `freeze_support()` / bootstrapping errors.
- A background command whose stdout is filtered through a pipe can look "empty" mid-run; write full
  output to a log file and inspect that, rather than trusting a grep'd tail.

**Implication for throughput.** Even with backgrounding, agent-driven is bounded by *attended* wall
time: the agent must be re-invoked to advance. That is the deliberate trade in §7. The mitigations above
(parallel fan-out + filler work + resumable file state) recover a lot of it, but if the goal becomes
"hundreds of iterations unattended overnight," the program-driven fallback (§7) is the right tool.

---

## Appendix — reproduce slice 1

```bash
PY=~/anaconda3/envs/rdagent/bin/python
$PY -m agent_loop.run_qlib --conf conf_baseline.yaml \
  --train_start 2015-01-01 --train_end 2016-12-31 \
  --valid_start 2017-01-01 --valid_end 2017-12-31 \
  --test_start 2018-01-01 --test_end 2018-12-31 --out /tmp/res.json
# Needs: Docker up, local_qlib:latest image, ~/.qlib/qlib_data/cn_data extracted.
# Does NOT need: .env, ANTHROPIC_API_KEY, or any LLM endpoint.
```
