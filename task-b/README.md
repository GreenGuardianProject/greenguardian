# Task B: scheduling

A discrete-event simulator replays a trace of 28,662 jobs from three sites in
15-minute ticks. At each tick the scheduler decides, for every waiting job,
whether to dispatch it now or hold it for a later bucket. The run is scored
against a first-come-first-served (FCFS) reference; higher is better.

This directory is the organisers' Task B starter kit (the `dirac_sim` package,
examples and tests) with our scheduler added.

## Where our code sits

| Path | Written by | What it is |
|---|---|---|
| `dirac_sim/schedulers/model_6_v4.py` | us | `Model6GreenWindowSchedulerV4`, our scheduler |
| `dirac_sim/schedulers/__init__.py` | us | Exports the scheduler |
| `examples/run_simulation.py` | organisers, changed by us | The runner used for all our results (see below) |
| `published_results/` | us (generated) | Score files and dispatch logs of our published runs |
| `dirac_sim/core/` | organisers | Simulator (`wms.py`), job queue, site model, scheduler interface, evaluator |
| `dirac_sim/baselines/` | organisers | `fcfs.py` and `greedy_carbon.py` |
| `dirac_sim/api/` | organisers | Forecast client and forecast API server |
| everything else | organisers | Backends, examples, tests, Docker files, docs |

## How our scheduler works

At start-up the scheduler loads our Task A forecast table,
`../task-a/model_6/outputs/forecast_submission.csv`, or the file named by the
`TASKB_FORECAST_CSV` environment variable. It also merges any forecasts the
simulator delivers at each tick; a delivered forecast replaces the preloaded
one for the same site and timestamp.

For each ready job, in deadline order (express jobs first), it compares the
best site available now with the best forecast bucket before the job's
deadline, allowing for slack. If the forecast bucket saves enough carbon
without costing much energy, the job is held and dispatched then; otherwise it
runs now. Every dispatch, immediate or deferred, is recorded in a forward
reservation ledger of slots, cores and memory per site and bucket, so that
held jobs do not collide when their bucket arrives. Jobs with too little slack,
and express jobs, run immediately. No job is held past the end of the
simulation window.

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

Installing `dirac_sim` from its own `pyproject.toml` (the `-e "./task-b[dev]"`
part) is what brings in `requests`. See the warning below.

## Run our scheduler

Run Task A first (`task-a/README.md`); the scheduler reads its output. Then,
from `task-b/`:

```bash
python examples/run_simulation.py --offline --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-13T17:00:00 \
  --scheduler dirac_sim.schedulers.Model6GreenWindowSchedulerV4 \
  --output-dir ../runs/task-b-full
```

This is the published run: final score 0.7317904319184824. With
`--end 2026-03-12T17:00:00` it is the truncated window, scoring
0.7332082197926377. `REPRODUCE.md` lists the expected totals for both and
explains why they differ.

Always use a new `--output-dir` for each run. The runner caches the FCFS
reference run as `baseline_execution_report_<start>_<end>.csv` in the output
directory, and on the next run into the same directory it loads that file
instead of re-simulating FCFS.

Use `examples/run_simulation.py`, not `python -m dirac_sim simulate`, for our
scheduler. The runner sets the `TASKB_EVAL_END` environment variable to the
window end, and the scheduler uses it to avoid holding jobs past the window.
The organisers' `python -m dirac_sim simulate` entry point does not set it,
so there the scheduler falls back to the earliest of the sites' last
forecast timestamps as the window end, and the results can differ.

## Run the baselines

```bash
python examples/run_simulation.py --offline --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-13T17:00:00 \
  --scheduler fcfs --output-dir ../runs/task-b-fcfs

python examples/run_simulation.py --offline --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-13T17:00:00 \
  --scheduler greedy_carbon --output-dir ../runs/task-b-greedy
```

`fcfs` and `greedy_carbon` are built-in names for
`dirac_sim.baselines.fcfs.FCFSScheduler` and
`dirac_sim.baselines.greedy_carbon.GreedyCarbonScheduler`. Without `--start`
and `--end`, the runner uses a 24-hour smoke window starting
`2025-11-19T23:00:00`. The organisers' entry point also runs the baselines:
`python -m dirac_sim simulate --offline --scheduler greedy_carbon`.

## Offline and live mode

The simulator feeds the scheduler a forecast bundle, with 1-hour and 24-hour
horizons, at every tick. The same bundles provide the per-site energy and
carbon signals that the simulator uses to charge each job.

- **Offline mode** (`--offline`) reads the bundles from a CSV file,
  `--forecast-csv`, by default `../data/forecast_baseline.csv`. No network or
  server is involved. All our published numbers are offline runs.
- **Live mode** (no `--offline`) requests the bundles over HTTP from a Task
  A-compatible service at `--api-url` (default `http://localhost:8000`),
  using `POST /forecast`. The organisers' Task A API
  (`uvicorn task_a.api:app`, see `task-a/README.md`) implements the contract.

### Warning: live mode without `requests` scores 0.0 silently

Live mode needs the `requests` package. `dirac_sim` declares it, so installing
`dirac_sim` from its own `pyproject.toml` provides it. But if you run
`dirac_sim` from an environment where it was not installed that way (for
example through `PYTHONPATH`, or the runner's own "run without installing"
path setup) and `requests` is missing, `dirac_sim/api/forecast_client.py`
does not fail. It returns an empty forecast bundle on every tick. Every site
signal then falls back to the simulator's default of 1000 Wh and 200 gCO2 per
bucket, FCFS and your scheduler are charged identically, and the run scores
0.0 with no error.

The symptom to look for is this line in the log, repeated on every tick:

```text
WARNING  dirac_sim.api.forecast_client  requests not installed; using empty bundle
```

If you see it, run `python -m pip install -e "./task-b[dev]"` from the
repository root.

### Warning: a missing forecast file also degrades silently

If `../task-a/model_6/outputs/forecast_submission.csv` does not exist and
`TASKB_FORECAST_CSV` is not set, the scheduler starts with no forecast table
and no warning. It then works only from the forecasts the simulator delivers,
and its results change. The file is committed to this repository. If you
delete it or regenerate it elsewhere, rerun Task A or set
`TASKB_FORECAST_CSV`.

## Our changes to `examples/run_simulation.py`

The runner is the organisers' file, with four changes we made during the
challenge. All our results were produced with it.

1. It sets `TASKB_EVAL_END` to the window end before building the scheduler.
2. It caches the FCFS reference run in the output directory, as described
   above, and still writes `baseline_execution_report.csv`.
3. It strips non-ASCII characters from the score summary it prints to the
   console, to avoid encoding errors on Windows consoles.
4. It writes `score_summary.txt` and `score_metrics.json` as UTF-8.

## Scoring

The evaluator (`dirac_sim/core/evaluator.py`) compares the scheduler's run
with the FCFS run over the same window. For energy, carbon and makespan
totals it computes a normalised improvement

```text
delta_x = (FCFS_x - ours_x) / FCFS_x          clamped to [-1, 1]
```

and then

```text
pareto  = 1 - (1/3) * ((1 - delta_energy) + (1 - delta_carbon) + (1 - delta_makespan))
bonus   = 0.15 * max(0, delta_declared)       delta of the --objective you declare
penalty = 0.005 * (percentage of jobs that missed their deadline)
final   = max(0, pareto + bonus - penalty)
```

`pareto` is also clamped to [0, 1]. Higher is better. FCFS scored against
itself gets 0.0. For our published run, delta_energy = 0.8912,
delta_carbon = 0.8994 and delta_makespan = 0, so pareto = 0.5969, the carbon
bonus is 0.1349, the penalty is 0, and the final score is 0.7318.

## Outputs

Each run writes to its `--output-dir`:

- `baseline_execution_report.csv` and `baseline_execution_report_<start>_<end>.csv`: the FCFS reference run
- `execution_report.csv`: the selected scheduler's execution records
- `dispatch_log.csv`: dispatch decisions
- `score_summary.txt`: human-readable score
- `score_metrics.json`: run settings, totals, deltas, penalties and final score

## Inputs

The simulator reads from the repository root `data/` directory:
`job_trace.csv` (the jobs), `site_config.json` (sites and capacities) and
`forecast_baseline.csv` (offline forecasts and site signals).
`python data/prepare_data.py` regenerates them from `data/raw_metrics/`.

## Writing a scheduler

Subclass `dirac_sim.core.scheduler.Scheduler` and return a `DispatchPlan`
from `schedule(queue, registry, forecast, now)`. The organisers' template is
`examples/custom_scheduler_template.py`. To defer a job, return a
`DispatchDecision` whose `dispatch_at` is in the future; jobs without a
decision stay pending until the next tick.

## Tests

```bash
cd task-b
python -m pytest
```

## Files the rules say not to modify

`dirac_sim/core/evaluator.py`, `dirac_sim/core/wms.py`,
`dirac_sim/core/job_queue.py`, `dirac_sim/core/site_model.py`,
`dirac_sim/core/scheduler.py`, `tests/`, and the input data under `../data/`.
None of them has been changed in this repository. (During the challenge we
ran with a locally modified `wms.py` that added a console progress bar and
faster job lookups. We have restored the organisers' version, and both
published runs reproduce with it to every digit.)

## Optional deployment

`docker-compose up --build` starts the organisers' API and monitoring demo.
