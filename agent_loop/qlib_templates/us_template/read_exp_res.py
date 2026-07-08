import pickle
from pathlib import Path

import pandas as pd
import qlib
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

qlib.init()

from qlib.workflow import R

# here is the documents of the https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: list all the recorder and metrics

# Assuming you have already listed the experiments
experiments = R.list_experiments()

# Iterate through each experiment to find the latest recorder
experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info["end_time"]
            try:
                # Check if the recorder has a valid end time
                if end_time is not None:
                    if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                        latest_recorder = recorder
                else:
                    print(f"Warning: Recorder {recorder_id} has no valid end time")
            except Exception as e:
                print(f"Error: {e}")

# Check if the latest recorder is found
if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder}")

    # Load the specified file from the latest recorder
    metrics = pd.Series(latest_recorder.list_metrics())

    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)

    print(f"Output has been saved to {output_path}")

    ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    ret_data_frame.to_pickle("ret.pkl")

    # ------------------------------------------------------------------------------------------------
    # Tier-2 extraction for the research-report UI (ADR 0003).
    #
    # The UI runs on the host, where qlib is NOT installed, so it cannot unpickle qlib `Position` objects
    # nor read the binary price data. Here — inside the container, with qlib available — we reduce both to
    # plain-pandas parquet the host can read directly:
    #   * positions.parquet : [datetime, instrument] -> amount, weight   (what the strategy actually held)
    #   * prices.parquet    : [datetime, instrument] -> close            (for the price + entry/exit plot)
    # Guarded end-to-end: any failure here leaves ret.pkl / metrics intact (Tier-1 figures still work).
    # ------------------------------------------------------------------------------------------------
    try:
        positions = latest_recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        rows = []
        items = positions.items() if hasattr(positions, "items") else enumerate(positions)
        for date, pos in items:
            pdict = getattr(pos, "position", pos)  # qlib Position -> .position dict (or already a dict)
            if not isinstance(pdict, dict):
                continue
            total = pdict.get("now_account_value")
            for inst, val in pdict.items():
                if inst in ("cash", "now_account_value") or not isinstance(val, dict):
                    continue
                amount = val.get("amount")
                if amount is None:
                    continue
                price = val.get("price")
                weight = val.get("weight")
                if weight is None and total and price is not None:
                    weight = amount * price / total
                rows.append({
                    "datetime": pd.Timestamp(date), "instrument": str(inst),
                    "amount": float(amount), "weight": float(weight) if weight is not None else None,
                })
        positions_df = pd.DataFrame(rows)
        if not positions_df.empty:
            positions_df.to_parquet("positions.parquet", index=False)
            print(f"[tier2] positions.parquet — {len(positions_df)} rows, "
                  f"{positions_df['instrument'].nunique()} names")

            # prices for the held names over the holding window
            from qlib.data import D

            insts = sorted(positions_df["instrument"].unique())
            start, end = positions_df["datetime"].min(), positions_df["datetime"].max()
            px = D.features(insts, ["$close"], start_time=start, end_time=end, freq="day")
            px = px.rename(columns={"$close": "close"}).reset_index()
            px.columns = [("datetime" if c in ("datetime", "level_1") else
                           "instrument" if c in ("instrument", "level_0") else c) for c in px.columns]
            px[["datetime", "instrument", "close"]].to_parquet("prices.parquet", index=False)
            print(f"[tier2] prices.parquet — {len(px)} rows")
        else:
            print("[tier2] no positions extracted; skipping positions/prices parquet")
    except Exception as e:  # noqa: BLE001 — Tier-2 is best-effort; never fail the run over a plot artifact
        print(f"[tier2] extraction skipped ({type(e).__name__}: {e})")
