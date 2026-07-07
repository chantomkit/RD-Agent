# Provisioning recent US data for the loop

The public qlib bundles (`cn_data`, prebuilt `us_data`) are **frozen ~2020**. To run the loop on recent
data (2024–2026) we build a fresh US dataset from Yahoo Finance. One-time recipe:

```bash
QP=~/anaconda3/envs/qlib/bin           # an env with qlib
$QP/pip install yfinance loguru

# 1. fetch adjusted daily OHLCV -> per-symbol CSVs (fixed large-cap universe + ^GSPC benchmark)
$QP/python -m agent_loop.data.build_us /tmp/us_csv

# 2. get qlib's dumper and convert CSVs -> qlib binary
curl -sSL -o /tmp/dump_bin.py https://raw.githubusercontent.com/microsoft/qlib/main/scripts/dump_bin.py
$QP/python /tmp/dump_bin.py dump_all --data_path /tmp/us_csv --qlib_dir ~/.qlib/qlib_data/us_data \
    --include_fields open,high,low,close,volume,factor --date_field_name date --symbol_field_name symbol --freq day

# 3. carve the tradeable universe out of the benchmark (SPX = benchmark, not a stock)
grep -vP '^SPX\t' ~/.qlib/qlib_data/us_data/instruments/all.txt > ~/.qlib/qlib_data/us_data/instruments/universe.txt
```

Then run the loop with the US templates:

```bash
python -m agent_loop.run_qlib --template_dir agent_loop/qlib_templates/us_template \
    --conf conf_baseline.yaml --train_start 2015-01-01 --train_end 2021-12-31 \
    --valid_start 2022-01-01 --valid_end 2023-12-31 --test_start 2024-01-01 --test_end 2024-12-31 --out /tmp/sel.json
```

## What the US templates change vs cn (`agent_loop/qlib_templates/us_template/`)
`provider_uri=us_data`, `region=us`, `market=universe` (102 large-caps), `benchmark=SPX`, no CN
price-limit (`limit_threshold` removed), concentrated `topk=30/n_drop=3` for the ~100-name universe.
Label and Alpha158 features are unchanged.

## Caveats (real, not hidden)
- **Survivorship bias:** the universe is a *fixed current* large-cap list, not point-in-time membership.
- **Adjusted prices:** `auto_adjust=True`, `factor=1.0` — good for research on adjusted close, not for
  exact fill modeling.
- **Holdout end buffer:** end the backtest a few weeks before the data edge; qlib's strategy peeks one
  trading day ahead to execute the final rebalance, so `test_end` = last calendar date raises IndexError.
- **Signal transfer:** Alpha158/Alpha20 were tuned for CN A-shares; expect weak cross-sectional signal on
  a small, efficient US large-cap universe. That is a research finding, not a bug — it's what the loop
  is for.
- **Knowledge substrate coupling:** `agent_loop/knowledge/` was extracted against `cn_data`. Field sets
  differ slightly (US has no `$change`; both lack `$vwap`). Regenerate for US if authoring field-sensitive
  factors: point `knowledge/build.py`'s `PROVIDER_URI` at `us_data`.
```
