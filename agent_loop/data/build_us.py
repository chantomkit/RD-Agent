"""
Provision recent US equity data for qlib (ADR 0002 — the US-market path).

The public qlib bundles (cn_data and the prebuilt us_data) are frozen ~2020. To run the loop on recent
data (2024-2026), we fetch adjusted daily OHLCV from Yahoo (yfinance) into per-symbol CSVs, then convert
to qlib binary format with qlib's `dump_bin.py`. See data/README.md for the full recipe.

Usage (in an env with yfinance + qlib, e.g. the `qlib` conda env):
    python -m agent_loop.data.build_us <csv_out_dir> [--tickers AAPL,MSFT,...] [--start 2015-01-01] [--end 2026-07-06]

Then dump:
    python dump_bin.py dump_all --data_path <csv_out_dir> --qlib_dir ~/.qlib/qlib_data/us_data \
        --include_fields open,high,low,close,volume,factor --date_field_name date --symbol_field_name symbol

Notes / caveats (documented, not hidden):
- auto_adjust=True gives split/dividend-adjusted OHLC; we set factor=1.0 (prices already adjusted) — fine
  for research backtests on adjusted close.
- Universe defaults to a fixed large-cap list (current membership) => SURVIVORSHIP BIAS. Documented.
- ^GSPC is stored as symbol SPX for use as the backtest benchmark; carve it out of the tradeable
  universe afterwards: `grep -vP '^SPX\\t' instruments/all.txt > instruments/universe.txt`.
"""
from __future__ import annotations

import time
from pathlib import Path

import fire
import pandas as pd

RENAME = {"^GSPC": "SPX"}
# A fixed liquid large-cap universe (used when no --tickers given). Current membership => survivorship bias.
DEFAULT_UNIVERSE = (
    "AAPL MSFT NVDA AMZN META GOOGL GOOG BRK-B LLY AVGO TSLA JPM V UNH XOM MA JNJ PG HD COST ORCL MRK "
    "ABBV CVX CRM BAC KO PEP WMT ADBE MCD CSCO ACN TMO ABT LIN DIS INTC WFC VZ QCOM TXN DHR NKE PM "
    "CMCSA NEE AMD RTX HON UNP IBM LOW SPGI CAT GE BA AXP GS ISRG BKNG PFE AMGN NOW SBUX BLK ELV GILD "
    "ADI T MDT SYK PLD C MMC DE LMT ADP CB VRTX MO TJX SCHW FISV REGN CI PGR BSX ZTS SO DUK BMY MU "
    "AMAT PANW CDNS SNPS KLAC MCO EQIX AON ITW SLB"
).split()


def build(csv_out: str, tickers: str | None = None, start: str = "2015-01-01", end: str = "2026-07-06",
          chunk: int = 40) -> None:
    import yfinance as yf

    out = Path(csv_out)
    out.mkdir(parents=True, exist_ok=True)
    tick = [t.strip() for t in tickers.split(",")] if tickers else list(DEFAULT_UNIVERSE)
    tick = sorted(set(tick)) + ["^GSPC"]
    print(f"[build_us] {len(tick)} symbols, {start}..{end}")

    written = 0
    for i in range(0, len(tick), chunk):
        batch = tick[i:i + chunk]
        data = yf.download(batch, start=start, end=end, interval="1d", auto_adjust=True,
                           progress=False, threads=True, group_by="ticker")
        for t in batch:
            try:
                sub = data if len(batch) == 1 else data[t]
            except KeyError:
                continue
            sub = sub.dropna(how="all")
            if sub.empty or len(sub) < 200:
                continue
            sym = RENAME.get(t, t)
            df = pd.DataFrame({
                "symbol": sym, "date": pd.to_datetime(sub.index).strftime("%Y-%m-%d"),
                "open": sub["Open"].values, "high": sub["High"].values, "low": sub["Low"].values,
                "close": sub["Close"].values, "volume": sub["Volume"].values, "factor": 1.0,
            }).dropna(subset=["open", "high", "low", "close"])
            if len(df) >= 200:
                df.to_csv(out / f"{sym}.csv", index=False)
                written += 1
        print(f"  {min(i + chunk, len(tick))}/{len(tick)} processed, {written} written", flush=True)
        time.sleep(2)
    print(f"[build_us] DONE: {written} symbols -> {out}")


if __name__ == "__main__":
    fire.Fire(build)
