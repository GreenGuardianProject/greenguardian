# Reproducing the published results

Run every command from the repository root unless a `cd` is shown, with the
environment active. Every run writes to a fresh directory under `runs/`,
which git ignores. A fresh directory matters for Task B: the runner caches its
FCFS baseline inside the output directory and reuses it on the next run.

Reference environment: Python 3.14.3, Windows 11, `requirements-lock.txt`. On
that combination every number below should match exactly. Other Python or
package versions may differ in the last digits.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e "./task-a[model,dev]" -e "./task-b[dev]"
python -m pip check
```

`pip check` should print `No broken requirements found.`

## 2. Test suites

```bash
(cd task-a && python -m pytest)
(cd task-b && python -m pytest)
```

## 3. Task A, our model

This runs first because Task B reads its output,
`task-a/model_6/outputs/forecast_submission.csv`.

```bash
(cd task-a && python model_6/train_model_6.py)
```

| Metric | Expected |
|---|---|
| `ScoreTA` | 0.6439136560442531 |
| `S_A_4` | 0.7183765582728562 |
| `S_A_96` | 0.8408457882937036 |
| `S_A1` | 0.2136796978360075 |
| `S_A2` | 0.4408925338037073 |

The script prints these and writes them to
`task-a/model_6/outputs/metrics_model_6.json`. The four output files are
committed, so `git status task-a/model_6/outputs` shows whether this run
reproduced them byte for byte.

## 4. Task A, reference baseline

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
cd ..
```

Expected `ScoreTA`: 0.7646450002485659.

## 5. Task B, our scheduler

Use `examples/run_simulation.py`, not `python -m dirac_sim simulate`. Only the
former passes the window end to our scheduler (see `task-b/README.md`).

Truncated window:

```bash
(cd task-b && python examples/run_simulation.py --offline --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-12T17:00:00 \
  --scheduler dirac_sim.schedulers.Model6GreenWindowSchedulerV4 \
  --output-dir ../runs/task-b-truncated)
```

Full window (the published run):

```bash
(cd task-b && python examples/run_simulation.py --offline --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-13T17:00:00 \
  --scheduler dirac_sim.schedulers.Model6GreenWindowSchedulerV4 \
  --output-dir ../runs/task-b-full)
```

Print the numbers to compare:

```bash
python -c "import json,sys; e=json.load(open(sys.argv[1]))['evaluation']; print(e['submission'], e['scores'])" runs/task-b-full/score_metrics.json
```

| Field | Truncated window | Full window |
|---|---|---|
| `jobs_dispatched` | 28642 | 28662 |
| `energy_wh_total` | 771810.49064897 | 791810.49064897 |
| `cfp_g_total` | 310547.75864442 | 314547.75864442 |
| `deadline_penalty` | | 0.0 |
| `final` | 0.7332082197926377 | 0.7317904319184824 |

The two runs are expected to differ. Twenty jobs arrive in the last two and a
half hours of the truncated window, and the scheduler is still holding them
for a cleaner slot when that window closes. With the extra day they all
dispatch, but after the end of the forecast signal table, so each is charged
the simulator's fallback of 1000 Wh and 200 gCO2. That is exactly the
20,000 Wh and 4,000 g difference.

The published run's score file and dispatch log are in
`task-b/published_results/full_window/`, and the truncated run's are in
`task-b/published_results/truncated_window/`.

## 6. Task B baselines

```bash
(cd task-b && python examples/run_simulation.py --offline --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-13T17:00:00 \
  --scheduler fcfs --output-dir ../runs/task-b-fcfs)
(cd task-b && python examples/run_simulation.py --offline --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-13T17:00:00 \
  --scheduler greedy_carbon --output-dir ../runs/task-b-greedy)
```

Both should run to completion. These are not published results; they are
what our verification run produced on the reference environment:

| Field | FCFS | Greedy-carbon |
|---|---|---|
| `jobs_dispatched` | 28662 | 28662 |
| `energy_wh_total` | 7279431.33108995 | 7315516.60570242 |
| `cfp_g_total` | 3127038.1921551498 | 3113136.21978814 |
| `final` | 0.0 | 0.0006668597333677781 |

FCFS scored against itself gives 0.0. Greedy-carbon logs a few hundred
"site full; job re-queued" warnings; that is expected.

## Run times

On the reference machine, Task A takes about a minute and the Task A
baseline a few seconds. Each Task B run takes about 8 to 11 minutes, because
every run also simulates the FCFS reference.
