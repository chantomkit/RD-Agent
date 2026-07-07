<!-- curated (grounded in the qlib templates + real run output). Update if run_qlib's result keys change. -->

# Metric glossary — what `run_qlib` returns and what the guardrail uses

`run_qlib` returns a metric Series (`exp.result`) written to the run JSON's `metrics`. The raw daily
portfolio series is in `<workspace>/ret.pkl` (`report_normal_1day`): daily excess return **with cost**
= `return - bench - cost`; **without cost** = `return - bench`.

## Signal quality (cross-sectional, per rebalance)
| metric | meaning |
|---|---|
| `IC` | mean Pearson corr(prediction, forward label) across days. |
| `Rank IC` | mean Spearman (rank) corr — more robust to outliers; the guardrail's default holdout metric. |
| `ICIR` / `Rank ICIR` | IC / Rank IC divided by its std — stability of the signal. |

## Portfolio performance (TopkDropout backtest; `1day.` prefix)
Each exists twice: `..._with_cost.*` (net of transaction cost) and `..._without_cost.*` (gross).
| metric | meaning |
|---|---|
| `1day.excess_return_with_cost.annualized_return` | annualized excess return over benchmark, **net of cost**. |
| `1day.excess_return_with_cost.information_ratio` | annualized Sharpe of the net excess return (a.k.a. **net IR**). |
| `1day.excess_return_with_cost.max_drawdown` | worst peak-to-trough of the net excess return. |
| `1day.excess_return_with_cost.{mean,std}` | daily mean / std of the net excess return. |
| `1day.excess_return_without_cost.*` | same, gross of cost. |

## What the guardrail reads (see guardrail.py)
- **Deflated Sharpe** is computed from the daily **with-cost** excess-return series in `ret.pkl`
  (real T, skew, kurtosis) — not from an annualized summary.
- **net_positive / beats_sota_net** use `1day.excess_return_with_cost.information_ratio` (net IR).
- **holdout gate** compares `Rank IC` on the locked holdout vs the SOTA's holdout `Rank IC`.

Rule of thumb: **cost matters.** A factor can look good gross (`without_cost`) yet lose net
(`with_cost`) if it is high-turnover. The guardrail rejects gross-only winners by design.
