"""COPY of Model6GreenWindowSchedulerV4 with per-site mean normalisation added to
_score_signal, to test the paper's claim (rank on c/c_bar_s, e/e_bar_s instead of
raw magnitudes). The original model_6_v4.py is untouched; this only overrides one
method."""
from __future__ import annotations

from dirac_sim.core.scheduler import Objective
from dirac_sim.schedulers.model_6_v4 import Model6GreenWindowSchedulerV4, _Signal


class Model6GreenWindowSchedulerV4Normalized(Model6GreenWindowSchedulerV4):
    def _site_means(self):
        cache = getattr(self, "_norm_means_cache", None)
        if cache is None:
            cache = {}
            for site_id, by_ts in self._signals.items():
                sigs = list(by_ts.values())
                n = len(sigs) or 1
                e_bar = sum(s.energy_wh for s in sigs) / n
                c_bar = sum(s.cfp_g for s in sigs) / n
                cache[site_id] = (e_bar if e_bar > 0 else 1.0,
                                  c_bar if c_bar > 0 else 1.0)
            self._norm_means_cache = cache
        return cache

    def _score_signal(self, signal: _Signal) -> tuple:
        e_bar, c_bar = self._site_means().get(signal.site_id, (1.0, 1.0))
        e = signal.energy_wh / e_bar
        c = signal.cfp_g / c_bar
        if self.declared_objective == Objective.ENERGY:
            return (e, c, signal.timestamp, signal.site_id)
        if self.declared_objective == Objective.MAKESPAN:
            return (0.60 * c + 0.40 * e, e, signal.timestamp, signal.site_id)
        return (0.80 * c + 0.20 * e, e, signal.timestamp, signal.site_id)
