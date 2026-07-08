# ADR 0003 — The research-report web UI: a read-only viewer over the loop's file state

- **Status:** Accepted
- **Date:** 2026-07-07
- **Deciders:** repo owner
- **Relationship to 0002:** **Extends [ADR 0002](0002-agent-driven-loop.md).** Adds a reporting/UI layer
  on top of 0002's file-based loop state. It changes **nothing** about the loop, the guardrail, or the
  execution path; it adds one new artifact (a per-loop `report.md`), optional lineage fields on the trace
  trial, a host-side read layer + Streamlit app, and a Tier-2 extraction so backtest plots can show real
  positions.

> Like 0001/0002, this is both a decision record and a cold-start handoff. A fresh session should be able
> to read this and continue. Paths are relative to the repo root.

---

## 1. TL;DR decision

After a loop iteration, ADR 0002's agent **concluded in the conversation** — so the results, the
selection-vs-holdout comparison, and the gate decision evaporated when the session ended. There was no
durable, browsable record and no way to point back at a past loop to fine-tune from it.

**Decision:** a **read-only Streamlit web app** — `python -m agent_loop.report_ui --port 19900` — that
browses **every** loop's report (gate decision, tabular comparison, backtest figures, and the agent's
discussion) over the state 0002 already writes to disk. The loop now **writes a `report.md`** at close
instead of only talking; the trace trial gains **lineage** (`parent_loop`) so the user can tell the agent
"branch from `us/1`" and the ledger tracks it. The web app **never triggers runs** — the loop stays
attended and agent-driven (0002); the browser is a viewer.

**Two forks, decided with the user:**
- **UI stack = Streamlit**, reusing RD-Agent's already-installed streamlit+plotly stack and its qlib
  `report_figure` helper (zero build step; matches the `rdagent ui --port …` ergonomics). *Not* the
  Vue/Flask `web/` SPA (npm build + Flask backend + adapters; its own README says it doesn't support the
  data_science scenario), and *not* a static-HTML export (no cross-loop browsing).
- **Interaction = read-only viewer; the agent branches.** No run-triggering from the browser.

Consequences: the whole UI is a **pure host-side reader** — **no qlib, no Docker, no `.env`** — because
every artifact it loads is plain pandas. The one thing that isn't (real positions) is reduced to plain
pandas *inside the container* by the existing post-run step.

---

## 2. For the next session — start here

**What is true right now (verified this session):**
- **The app is built and renders end-to-end.** `agent_loop/report_ui.py` (launcher) →
  `agent_loop/report_app.py` (Streamlit) over `agent_loop/report.py` (the read layer). Verified headless
  via `streamlit.testing.v1.AppTest` against the **real** US + CN ledgers: gate panel, sel-vs-holdout
  table, baseline floor, all figures, both views (loop detail + project trends), every loop — **no
  exceptions, no qlib on the host.**
- **Tier-1 figures are free** (no new pipeline): `report_figure(ret.pkl)` for the cumulative-return
  family, plus IC/Rank-IC time series + signal-quantile bars (from `pred.pkl`+`label.pkl`), cumulative
  long-short spread (`long_short_r.pkl`), and portfolio value/turnover — all load as plain pandas.
- **Tier-2 (real price + entry/exit) works.** The US template's `read_exp_res.py` now also dumps
  host-readable `positions.parquet` + `prices.parquet`. Verified by running the extraction in-container
  against a real workspace (7530 position rows / 219 names, weights ≈ 1/30 for topk=30; 54854 price rows)
  and then rendering NVDA's price with 5 entry / 5 exit marks **on the host with no qlib**.
- **Lineage** is in `trace.py` (`record(..., parent=, title=, report=, run_dir=)`), backward-compatible.
- Tests: `agent_loop/tests/test_report.py` (16) + a lineage test in `test_trace.py`. Full suite **34
  passing**.

**Environment facts (this machine, verified):** streamlit 1.59 + plotly 6.8 + pyarrow 24 are in the
`rdagent` conda env (`~/anaconda3/envs/rdagent/bin/python`). qlib is **not** in that env (it lives only in
the `local_qlib:latest` Docker image) — which is *why* the UI must stay plain-pandas. Run the app with
`~/anaconda3/envs/rdagent/bin/python -m agent_loop.report_ui --port 19900`.

**Now open (future, not blockers):** wire the Tier-2 extraction into the CN/model path too (today it's
US-only because the CN/model templates live under `rdagent/`, kept untouched — §5); a holdings
concentration/heatmap panel (data is already in `positions.parquet`); and richer report authoring
conventions as more loops accrue.

---

## 3. Architecture: who does what

| Concern | Where | Notes |
|---|---|---|
| Loop state (ledger, run/decision JSON, workspaces) | ADR 0002, on disk | unchanged |
| Per-loop **narrative** | **agent** writes `runs[_us]/loop_N/report.md` | replaces the in-chat conclusion |
| **Lineage** | `trace.record(parent=…)` → trial `parent_loop` | for "branch from loop N" |
| **Read layer** | `agent_loop/report.py` | pure pandas; no qlib/Streamlit; unit-tested |
| **UI** | `agent_loop/report_app.py` (Streamlit) + `report_ui.py` (launcher) | presentation only |
| **Tier-2 extraction** | `qlib_templates/us_template/read_exp_res.py` | in-container; dumps parquet |

The read layer is the contract; the Streamlit file is thin presentation, so the logic is testable without
a browser (and was — see §6).

---

## 4. What's new (kept minimal)

The structured numbers already exist (ledger trial + run/decision JSON, incl. the guardrail's
human-readable `reasons`). So the additions are deliberately small:

1. **`report.md` per loop** — the agent's discussion. The only genuinely new artifact.
2. **Lineage fields on the trial** — `parent_loop`, `title`, `report_path`, `run_dir`, all optional and
   stored only when provided (pre-0003 ledgers load unchanged).
3. **Two Tier-2 parquets** — `positions.parquet`, `prices.parquet` — dumped by the US extraction step so
   the price+entry/exit plot has real positions without putting qlib on the host.

---

## 5. The figures — split by the host constraint (the key finding)

qlib is not installed on the host (0002 keeps it in Docker), so a workspace artifact is renderable on the
host **iff** it loads as plain pandas. That single fact partitions the catalog:

**Tier 1 — plain pandas, no qlib, no new pipeline** (every run already produces these):
- **Cumulative return** (own, clearly-labelled) — strategy *absolute* (after/before cost), *benchmark*,
  and *excess* on one axis, plus separate **drawdown** and **turnover** charts, all from `ret.pkl`. We
  keep the reused `rdagent/log/ui/qlib_report_figure.report_figure` (its inputs are exactly `ret.pkl`'s
  columns) but tuck it in a collapsed "detailed" expander — it is rich but dense.
- **IC & Rank-IC time series**, **signal-quantile forward-return bars** — from `pred.pkl` + `label.pkl`.
  (Cross-check: the computed mean Rank IC equals the ledger's stored selection Rank IC.)
- **Cumulative long-short spread** — from `sig_analysis/long_short_r.pkl`.
- **Portfolio value / cash** (absolute account, mark-to-market) — from `ret.pkl`.

**Presentation (from review feedback):** every figure carries a title + x/y-axis labels; metric names are
spelled out (IC/ICIR/IR) with a plain-language glossary; and because the return metrics are *excess vs.
benchmark*, an **Absolute vs. benchmark** table (strategy / benchmark / excess, from the equity curve)
sits under the metrics — so a positive excess in a falling market (an absolute account loss) reads
clearly instead of looking like a contradiction.

**Backtest-setup panel (from review):** each report opens with the full setup — universe, **benchmark**
(the market index the excess/IR are measured against; e.g. `SPX` = S&P 500, also the `buy_hold`
baseline — *not* a buy-and-hold of the strategy), data region, model, feature base, the TopkDropout
strategy knobs, per-side costs, and the train / valid / **selection** / **locked-holdout** date windows.
It is parsed from the run's qlib config (`report.backtest_setup`; regex handles both the CN concrete
values and the US jinja `default(...)` forms), with any explicit `run_qlib` overrides (now recorded in
the run JSON `settings`) taking precedence. The Tier-2 price+entry/exit plot is **per-segment** — the
Selection and Holdout tabs each show their own run's trades over their own window (extraction runs on
every backtest, so both are populated going forward).

**"How the strategy works" explainer (from review):** because a one-line hypothesis
("add MOM12_1 + VOL60 to Alpha20") doesn't convey the *mechanism*, each report also carries a generated
plain-English walk-through (`report.strategy_explainer`): what the model predicts (the label decoded from
the config — e.g. next-day close-to-close forward return, offset to avoid look-ahead), how positions are
**entered** (rank by score, hold the top-k) and **exited** (Top-k Dropout: sell the `n_drop` worst-held
each day, no stop-loss), the resulting turnover, execution/cost, and how it's scored vs. the benchmark.
It ties the ▲/▼ marks on the price charts to the actual entry/exit rule. Generated from the config so it
is always present and consistent; the agent's `report.md` adds the *specific* rationale on top.

**Tier 2 — real positions + price** (`positions_normal_1day.pkl` holds qlib `Position` objects → cannot
unpickle on the host; the binary price data also needs qlib). Resolved by extracting **where qlib already
lives**: the in-container post-run step (`read_exp_res.py`, the same one that emits `ret.pkl`) now also
writes `positions.parquet` + `prices.parquet` as plain pandas. Then the host draws a **ticker → price line
with ▲entry/▼exit marks** with zero qlib. The extraction is fully guarded — any failure leaves Tier-1
intact.

**Scope honesty:** Tier-2 lands on the **US path** (the template we own). The built-in CN/model templates
live under `rdagent/` and are kept untouched (0002's cheap-upstream-merge rule), so Tier-2 degrades
gracefully there (no `positions.parquet` → those panels are skipped) while Tier-1 works for every run.

---

## 6. Verification

- **Unit** (`~/anaconda3/envs/rdagent/bin/python -m pytest agent_loop/tests/ -q`): 34 passing. The report
  tests fabricate a ledger + workspace artifacts in `tmp_path` and exercise assembly, graceful
  degradation, the IC/quantile builders (IC positive for an injected-correlation signal), and the Tier-2
  entry/exit interval logic.
- **App** (headless `AppTest` against the real ledgers): renders every loop + both views with no
  exception. `AppTest` caught a real `StreamlitDuplicateElementId` bug (identical charts in the
  selection/holdout tabs) — fixed with per-chart `key=`.
- **Tier-2** (in-container extraction against a real workspace → host render): produced valid parquet and
  a correct NVDA entry/exit plot on the host.
- To view live: `~/anaconda3/envs/rdagent/bin/python -m agent_loop.report_ui --port 19900`, open
  `http://127.0.0.1:19900`. No Docker, no `.env`.

---

## 7. What we deliberately don't do

- **No run-triggering from the browser.** The loop is attended and agent-driven (0002 §7); the app is a
  viewer. "Fine-tune from a report" = the user reads a report id (`us/1`) and tells the agent to branch;
  the ledger records `parent_loop`. This keeps the agent as the brain and avoids a Docker-executing web
  backend.
- **No Vue/Flask SPA, no `LoopBase` adapter.** We read our own JSON directly rather than adapting
  RD-Agent's log-stream/`Message`/`Hypothesis` machinery (which 0002 avoided). The Streamlit app is the
  smallest thing that meets the need with the most reuse.

---

## Appendix — run the UI

```bash
PY=~/anaconda3/envs/rdagent/bin/python
$PY -m agent_loop.report_ui --port 19900          # then open http://127.0.0.1:19900
# --root defaults to git_ignore_folder/agent_loop (where the trace*.json ledgers live)
# Needs: streamlit/plotly/pyarrow in the rdagent env. Does NOT need Docker, qlib, or a .env.
```
