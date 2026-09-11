# GreenGuardian: GreenDIGIT Discovery Challenge, ECML-PKDD 2026

This is our entry to the GreenDIGIT Discovery Challenge at ECML-PKDD 2026, by
Nikolaos Vatopoulos and Dimitrios Thanasoulias (University of Thessaly). The
method is described in our paper, *Asymmetric Gradient Boosting for Carbon
Forecasting and Deadline-Safe Green-Window Scheduling*
(`paper/GreenGuardian_Paper.pdf`).

The repository is the organisers' starter kit with our code added. The
Attribution section below says which parts are theirs and which are ours.

## The two tasks

The dataset covers three anonymised grid-computing sites, metered into
15-minute buckets over about 3.7 months (November 2025 to March 2026).

**Task A, forecasting.** Predict each site's energy (`energy_wh`) and carbon
footprint (`cfp_g`) 1 hour (4 steps) and 24 hours (96 steps) ahead, and
classify each bucket for two events: A.1, missing or invalid signal, and A.2,
peak. The score is `ScoreTA`; lower is better.

Our approach trains four direct LightGBM models (energy and carbon at each
horizon) on a custom asymmetric pseudo-Huber loss and averages three seeds.
The carbon forecast leans on the energy forecast through a per-site
carbon-per-energy ratio. See `task-a/README.md`.

**Task B, scheduling.** A discrete-event simulator replays a trace of 28,662
jobs, and the scheduler decides when each job runs. The score compares the
run against a first-come-first-served (FCFS) reference; higher is better.

Our scheduler, `Model6GreenWindowSchedulerV4`, holds jobs for buckets that our
Task A forecast predicts to be lower-carbon, within each job's deadline
slack. It keeps a forward reservation ledger over slots, cores and memory so
that deferred jobs do not collide. See `task-b/README.md`.

## Results

| | Ours | Reference |
|---|---|---|
| Task A, `ScoreTA` (lower is better) | 0.6439 | 0.7646 (organisers' baseline) |
| Task B, final score (higher is better) | 0.7318 | 0.0 (FCFS, by construction) |

On Task B, against FCFS, our scheduler reduces carbon by 89.9% and energy by
89.1%, with makespan unchanged and all 28,662 deadlines met. `REPRODUCE.md`
has the commands and the exact values.

## Repository layout

```text
data/                    organisers' dataset: raw metrics, job trace, site config, offline forecasts
task-a/                  organisers' Task A starter kit
  model_6/               ours: Task A model and its submitted outputs
task-b/                  organisers' Task B simulator, evaluator and baselines
  dirac_sim/schedulers/  ours: the Task B scheduler
  published_results/     score files and dispatch logs of our published runs
experiments/             the per-site-normalised ranking variant discussed in the paper
paper/                   the paper
  figures_data/          source data for the paper's figures (no script in this repository generates it)
static/                  organisers' logos
requirements-lock.txt    exact package versions used for the published numbers
REPRODUCE.md             commands and expected numbers
```

## Quickstart

```bash
git clone <repository-url> greenguardian
cd greenguardian
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e "./task-a[model,dev]" -e "./task-b[dev]"
cd task-a
python model_6/train_model_6.py
```

On Windows, if `python` is not on your PATH, create the environment with
`py -3.14 -m venv .venv`. Once the environment is active, `python` works.

The last command trains the Task A model, rewrites `task-a/model_6/outputs/`,
and prints the metrics, ending with `"ScoreTA": 0.6439136560442531`. Task B is
covered in `task-b/README.md`, and the full check in `REPRODUCE.md`.

## Reproducibility

The published numbers were produced on Python 3.14.3 (Windows 11, 64-bit)
with the versions pinned in `requirements-lock.txt`. On that combination the
numbers in `REPRODUCE.md` should match to all 16 digits. Other Python or
package versions may change the last digits, mainly through LightGBM and
floating-point summation order. That does not by itself indicate a problem.

## One environment instead of two

The starter kit's instructions create a separate virtual environment inside
`task-a/` and another inside `task-b/`. That layout still works: install
`task-a[model,dev]` into one and `task-b[dev]` into the other. We document a
single environment at the repository root instead, for two reasons. Task B
reads Task A's output (`task-a/model_6/outputs/forecast_submission.csv`), so
the two tasks are run together. And the machine we used has a single Python
interpreter. pip resolves the two packages together without conflict.

## Attribution

The organisers' starter kit and dataset come from
<https://github.com/GreenDIGIT-project/greendigit-ecml-pkdd-2026-challenge>
(challenge website: <https://gd2.lab.uvalight.net/>). That includes everything
under `data/`, `static/`, `task-a/` other than `model_6/`, and `task-b/`
other than our scheduler and published results. This covers the simulator,
the evaluators, the submission validators, the reference baselines and the
test suites.

Our code is three files:

- `task-a/model_6/train_model_6.py`
- `task-a/model_6/lgbm_core.py`
- `task-b/dirac_sim/schedulers/model_6_v4.py`

We also wrote the scheduler package's `__init__.py`,
`experiments/model_6_v4_normalized.py`, and the documentation. We changed one
organisers' file, `task-b/examples/run_simulation.py`; `task-b/README.md`
describes the change.

The starter kit and dataset were funded by the European Union's Horizon
Europe research and innovation programme through the
[GreenDIGIT project](https://greendigit-project.eu/), grant agreement
No. [101131207](https://cordis.europa.eu/project/id/101131207).

<div style="display:flex;align-items:center;width:100%;">
  <img src="static/EN-Funded-by-the-EU-POS-2.png" alt="EU Logo" width="250px">
  <img src="static/cropped-GD_logo.png" alt="GreenDIGIT Logo" width="110px" style="margin-right:100px">
</div>

## Licence

MIT; see `LICENSE`. The starter kit's own `pyproject.toml` files also declare
MIT. The dataset belongs to the challenge organisers.

## Citation

Nikolaos Vatopoulos and Dimitrios Thanasoulias. *Asymmetric Gradient Boosting
for Carbon Forecasting and Deadline-Safe Green-Window Scheduling.* GreenDIGIT
Discovery Challenge, 1st Place, ECML-PKDD 2026.
