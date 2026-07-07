<!-- curated (grounded in rdagent/scenarios/qlib/experiment/*/conf_*.yaml). Update if the templates change. -->

# Model cards — what `run_qlib --conf <template>` trains

The model consumes the feature set (base handler + any factors) and predicts the 2-day forward label.
Pick the conf template by what you're testing.

## Factor templates (`--template factor`, the default)
| conf | model | features | use for |
|---|---|---|---|
| `conf_baseline.yaml` | **LGBModel** (gradient-boosted trees) | Alpha158-style base features (the `--features` JSON; default `ALPHA20`) | testing a **factor** (via `--features` expressions) against the LGBM baseline. |
| `conf_combined_factors.yaml` | **LGBModel** | base features **+** a `combined_factors_df.parquet` (`--factors`) | testing a factor supplied as a computed **parquet** (arbitrary Python). |
| `conf_combined_factors_sota_model.yaml` | the SOTA PyTorch `model.py` | base + combined factors | evaluating a factor on top of a promoted custom model. |

### LGBModel default hyperparameters (baseline)
`loss=mse, learning_rate=0.2, colsample_bytree=0.8879, subsample=0.8789, lambda_l1=205.7,
lambda_l2=580.98, max_depth=8, num_leaves=210`. Deterministic given features — good for isolating a
factor's marginal contribution.

## Model templates (`--template model`)
| conf | dataset | use for |
|---|---|---|
| `conf_baseline_factors_model.yaml` | `DatasetH` (Tabular) or `TSDatasetH` (TimeSeries) | testing a **model** on the base factors. |
| `conf_sota_factors_model.yaml` | same | testing a model on top of SOTA factors. |

### PyTorch `model.py` interface (what the model coder writes)
- Define `class Net(nn.Module)` — hard-coded as `model.py:Net` in the config.
- `model_type` ∈ {`Tabular` → `DatasetH`, `TimeSeries` → `TSDatasetH` with `step_len=20`, `num_timesteps=20`}.
- `training_hyperparameters` (env-injected): `n_epochs` (default 100), `lr` (2e-4), `early_stop` (10),
  `batch_size` (256), `weight_decay` (1e-4).
- Input width = number of features (`num_features`); for TimeSeries the input is `[batch, step_len, num_features]`.

## Guidance
- To test a **factor idea**, keep the model fixed (LGBM baseline) and vary `--features` / `--factors` —
  this isolates the factor's marginal value and is cheap.
- To test a **model idea**, keep features fixed and vary the model. Costlier (GPU training); budget it.
