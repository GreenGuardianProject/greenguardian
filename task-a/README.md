# Task A: forecasting

Task A asks for forecasts of each site's energy use (`energy_wh`) and carbon
footprint (`cfp_g`) 1 hour ahead (h = 4 fifteen-minute steps) and 24 hours
ahead (h = 96), plus two per-bucket classifiers:

- A.1 flags buckets whose signal is missing or invalid.
- A.2 flags peak events.

This directory is the organisers' Task A starter kit with our model added in
`model_6/`.

## Where our code sits

| Path | Written by | What it is |
|---|---|---|
| `model_6/train_model_6.py` | us | Entry point: trains, writes the three submission CSVs, scores them |
| `model_6/lgbm_core.py` | us | Feature engineering, custom objective, LightGBM training |
| `model_6/outputs/` | us (generated) | The submitted CSVs and `metrics_model_6.json` |
| `src/task_a/` | organisers | Data loading, labels, validators, evaluator, reference baseline, CLI, forecast API |
| `scripts/`, `config/`, `tests/` | organisers | Helper scripts, configuration, test suite |
| `outputs/` | organisers | Their committed reference-baseline outputs |

`train_model_6.py` imports the organisers' `task_a.dataio`, `task_a.labels`,
`task_a.submission` and `task_a.evaluation`, so our files are validated and
scored by the official code, unchanged.

## Install

Use the single environment described in the root README. From the repository
root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e "./task-a[model,dev]" -e "./task-b[dev]"
```

The `model` extra declares `lightgbm`, `numpy` and `pandas`. Our model needs
them; the starter kit alone does not.

## Run our model

```bash
cd task-a
python model_6/train_model_6.py
```

The script reads `../data/raw_metrics/summary_sites_15m.csv`, trains on rows
at or before the official cutoff `2026-02-18T14:00:00+00:00`, predicts the
rows after it, writes four files to `model_6/outputs/`, and prints the
metrics. Options: `--input`, `--output-dir`, `--seed` (default 2026).

Expected result: `ScoreTA = 0.6439136560442531`. The component scores are in
`REPRODUCE.md`.

Task B's scheduler reads `model_6/outputs/forecast_submission.csv`. If you
write the outputs elsewhere with `--output-dir`, point Task B at the new file
with the `TASKB_FORECAST_CSV` environment variable.

### How the model works

- Four direct LightGBM models, one per target (`energy_wh`, `cfp_g`) and
  horizon (4, 96). Each predicts `log1p` of the target h steps ahead from
  features known at the forecast origin: time-of-day and day-of-week
  encodings, lags of energy, carbon and core count (1 to 672 steps), rolling
  means, standard deviations and quantiles over 3 and 24 hours, per-core
  ratios, and a site code.
- A custom objective: an asymmetric pseudo-Huber loss on the `log1p` target
  (delta 0.18, under-prediction weighted 1.06 and over-prediction 0.94), used
  as a smooth stand-in for sMAPE.
- Each model is trained with three seeds (`seed`, `seed + 42`,
  `seed + 999`) and the predictions are averaged.
- The carbon forecast is a weighted blend of the direct carbon model and the
  energy forecast multiplied by the site's carbon-per-energy ratio at the
  forecast origin, with the site's training-period median ratio as fallback.
- A.1 score per bucket: a fixed blend of the site's training-period validity
  rate, the validity rate for that quarter-hour of the day, and the global
  rate.
- A.2 score per bucket: the forecast signal relative to the site's
  training-period 95th-percentile peak threshold, with the decision cutoff
  calibrated on training data.

## Run the reference baseline

These are the organisers' commands, writing to `../runs/` so the organisers'
committed `outputs/` stay untouched:

```bash
cd task-a
mkdir -p ../runs/task-a-baseline
task-a train-baseline --input ../data/raw_metrics/summary_sites_15m.csv \
  --model ../runs/task-a-baseline/baseline_model.json
task-a predict --input ../data/raw_metrics/summary_sites_15m.csv \
  --model ../runs/task-a-baseline/baseline_model.json \
  --output ../runs/task-a-baseline/forecast_submission.csv
task-a predict-detection --input ../data/raw_metrics/summary_sites_15m.csv \
  --output ../runs/task-a-baseline/detection_submission.csv
task-a predict-peaks --input ../data/raw_metrics/summary_sites_15m.csv \
  --output ../runs/task-a-baseline/peak_submission.csv
task-a evaluate --input ../data/raw_metrics/summary_sites_15m.csv \
  --forecasts ../runs/task-a-baseline/forecast_submission.csv \
  --detections ../runs/task-a-baseline/detection_submission.csv \
  --peaks ../runs/task-a-baseline/peak_submission.csv \
  --output ../runs/task-a-baseline/metrics.json
```

Expected result: `ScoreTA = 0.7646450002485659`.

Read the baseline's score with care. Its A.1 and A.2 files are built from the
true labels of the observed test rows (`cmd_predict_detection` and
`cmd_predict_peaks` in `src/task_a/cli.py`), so it scores a perfect
`S_A1 = S_A2 = 0`. Only its forecasting part is a real baseline. On
forecasting alone the comparison is `S_A_4` 1.0909 against our 0.7184, and
`S_A_96` 1.0938 against our 0.8408.

## Output files

`forecast_submission.csv`:

```text
series_id,forecast_timestamp_utc,horizon_steps_15m,energy_wh_pred,cfp_g_pred
```

`horizon_steps_15m` is 4 or 96. Values must be finite and non-negative, and
each `(series_id, forecast_timestamp_utc, horizon_steps_15m)` key may appear
once.

`detection_submission.csv` (A.1):

```text
series_id,bucket_15m,valid_signal_score,valid_signal_pred
```

`peak_submission.csv` (A.2):

```text
series_id,bucket_15m,peak_score,peak_pred
```

For both classification files, scores are in `[0, 1]`, predictions are 0 or
1, and each `(series_id, bucket_15m)` key may appear once. For all three
files, evaluation fails if any row required by the official split is missing.

Check a set of files with `task-a validate-submission --forecasts ...`,
`task-a validate-detection --detections ...` and
`task-a validate-peaks --peaks ...`.

## Scoring

Lower is better. The evaluator (`src/task_a/evaluation.py`) computes:

```text
S_A_h  = 0.5 * sMAPE(energy) + 0.5 * sMAPE(cfp)      for h = 4 and h = 96
S_A1   = 1 - 0.5 * (AUROC + F1)                      for A.1
S_A2   = 1 - 0.5 * (AUROC + F1)                      for A.2
ScoreTA = 0.7 * mean(S_A_4, S_A_96) + 0.15 * S_A1 + 0.15 * S_A2
```

Forecast error carries 70% of the weight; each classifier carries 15%.

## Forecast API

The organisers' API serves a Task A model over the `/forecast` contract that
Task B's live mode calls:

```bash
TASK_A_MODEL=../runs/task-a-baseline/baseline_model.json \
  uvicorn task_a.api:app --host 0.0.0.0 --port 8000
```

## Tests

```bash
cd task-a
python -m pytest
```

## Files the rules say not to modify

`src/task_a/evaluation.py`, `src/task_a/submission.py`,
`src/task_a/schemas.py`, `config/eval.yaml`, `tests/`, and the input data under
`../data/raw_metrics/`. None of them has been changed in this repository.
