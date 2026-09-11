# Experiments

## `model_6_v4_normalized.py`

`Model6GreenWindowSchedulerV4Normalized` is a subclass of the submitted Task B
scheduler (`task-b/dirac_sim/schedulers/model_6_v4.py`). It overrides one
method, `_score_signal`, so that candidate sites and time slots are ranked on
each forecast signal divided by that site's mean forecast (`c / c_bar_s`,
`e / e_bar_s`) instead of on raw magnitudes.

Our paper describes the ranking with this per-site mean normalisation. The
scheduler we actually submitted ranks on raw magnitudes. This variant exists
to show what that difference costs: on the truncated window
(`2025-11-19T23:00:00` to `2026-03-12T17:00:00`) it scores 0.7330, against
0.7332 for the submitted scheduler. The recorded score file is
`task-b/published_results/normalized_ranking/score_metrics.json`.

To run it, from `task-b/` with the environment from the root README active:

```bash
PYTHONPATH=../experiments python examples/run_simulation.py --offline \
  --objective carbon \
  --start 2025-11-19T23:00:00 --end 2026-03-12T17:00:00 \
  --scheduler model_6_v4_normalized.Model6GreenWindowSchedulerV4Normalized \
  --output-dir ../runs/task-b-normalized
```

On Windows PowerShell, set the path with `$env:PYTHONPATH = "..\experiments"`
before running the same command.
