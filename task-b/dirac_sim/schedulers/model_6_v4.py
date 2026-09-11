"""
V4 Scheduler for Task B using Model 6 forecasts.
This scheduler accurately tracks capacity over the duration of the jobs to maximize hold times
without causing site overload.
"""
from __future__ import annotations

import csv
import logging
import os
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dirac_sim.core.job_queue import Job, JobQueue, Priority
from dirac_sim.core.scheduler import (
    DispatchDecision,
    DispatchPlan,
    ForecastBundle,
    Objective,
    Scheduler,
)
from dirac_sim.core.site_model import Site, SiteRegistry

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _Signal:
    timestamp: datetime
    site_id: str
    energy_wh: float
    cfp_g: float


def _bucket_start(t: datetime) -> datetime:
    minutes = (t.minute // 15) * 15
    return t.replace(minute=minutes, second=0, microsecond=0)


class Model6GreenWindowSchedulerV4(Scheduler):
    """
    V4 Scheduler:
    - Maximize lookahead (up to deadline).
    - Exact per-bucket duration reservation tracking to prevent site overload.
    - Strong preference for green windows.
    """

    def __init__(
        self,
        declared_objective: str = "carbon",
        max_hold_hours: float = 72.0,  # Huge lookahead, capped by deadline
        carbon_saving_threshold: float = 0.01,
        energy_saving_threshold: float = 0.01,
        min_slack_minutes: float = 15.0,
        bucket_headroom: float = 1.0,  # use up to 100% of max capacity
        forecast_csv: Optional[str] = None,
    ) -> None:
        obj = Objective(declared_objective) if isinstance(declared_objective, str) else declared_objective
        super().__init__(
            declared_objective=obj,
            delta_energy_budget=0.05,
            delta_carbon_budget=0.05,
            delta_makespan_budget=0.05,
        )
        self.max_hold = timedelta(hours=max_hold_hours)
        self.carbon_saving_threshold = carbon_saving_threshold
        self.energy_saving_threshold = energy_saving_threshold
        self.min_slack = timedelta(minutes=min_slack_minutes)
        self.bucket_headroom = bucket_headroom

        self._signals: dict[str, dict[datetime, _Signal]] = {}
        self._timeline: dict[str, list[_Signal]] = {}
        self._timestamps: dict[str, list[datetime]] = {}

        # site_id -> bucket -> dict of resources
        self._reservations: Dict[str, Dict[datetime, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {'slots': 0, 'cores': 0, 'memory': 0.0})
        )

        self._eval_end = self._read_eval_end()
        self._load_forecast_csv(forecast_csv or self._default_forecast_csv())
        
        self._cleanup_time = None

    def on_forecast_received(self, bundle: ForecastBundle) -> None:
        added = False
        for signal in self._iter_signals(bundle.horizon_1h, bundle.horizon_24h):
            self._signals.setdefault(signal.site_id, {})[signal.timestamp] = signal
            added = True
        if added:
            self._rebuild_timeline()
            
    def on_tick_start(self, now: datetime) -> None:
        # Cleanup old reservations
        if self._cleanup_time is None or now - self._cleanup_time > timedelta(hours=1):
            threshold = now - timedelta(hours=2)
            for site_id in self._reservations:
                old_keys = [t for t in self._reservations[site_id] if t < threshold]
                for t in old_keys:
                    del self._reservations[site_id][t]
            self._cleanup_time = now

    def schedule(
        self,
        queue: JobQueue,
        registry: SiteRegistry,
        forecast: ForecastBundle,
        now: datetime,
    ) -> DispatchPlan:
        plan = DispatchPlan(
            declared_objective=self.declared_objective,
            delta_energy_budget=self.delta_energy_budget,
            delta_makespan_budget=self.delta_makespan_budget,
        )

        site_available_now: Dict[str, int] = {
            site.site_id: site.capacity.available_slots
            for site in registry.all_sites()
        }

        ready_jobs = list(queue.ready_jobs(now))
        # Sort by deadline ascending, then priority descending (EXPRESS first)
        ready_jobs.sort(key=lambda j: (j.deadline, j.priority != Priority.EXPRESS))

        for job in ready_jobs:
            decision = self._decide(job, registry, now, site_available_now)
            if decision is not None:
                plan.add(decision)
                if decision.dispatch_at <= now + timedelta(seconds=1):
                    cap = site_available_now.get(decision.site_id, 0)
                    site_available_now[decision.site_id] = max(0, cap - 1)

        return plan

    def _decide(
        self,
        job: Job,
        registry: SiteRegistry,
        now: datetime,
        site_available_now: Dict[str, int],
    ) -> Optional[DispatchDecision]:
        candidates = [
            site
            for site in registry.available_sites(job.site_whitelist)
            if site.capacity.max_concurrent_jobs > 0
        ]
        
        # Filter for immediate dispatch capacity
        immediate_candidates = [
            site for site in candidates 
            if site_available_now.get(site.site_id, 0) > 0 and self._has_capacity(site, job)
        ]
        
        if not immediate_candidates and not candidates:
            return None

        current = None
        if immediate_candidates:
            current = min(
                (
                    _Signal(
                        timestamp=now,
                        site_id=site.site_id,
                        energy_wh=site.get_energy(now),
                        cfp_g=site.get_carbon(now),
                    )
                    for site in immediate_candidates
                ),
                key=self._score_signal,
            )

        remaining_slack = job.deadline - now
        effective_max_hold = min(self.max_hold, remaining_slack * 0.95)
        
        latest = min(job.deadline, now + effective_max_hold)
        if self._eval_end is not None:
            latest = min(latest, self._eval_end)

        too_little_slack = (latest <= now) or (latest - now < self.min_slack)
        
        latency_val = candidates[0].dispatch_latency if candidates else 2.0
        fallback_latency = timedelta(minutes=latency_val)
        
        if too_little_slack or job.priority == Priority.EXPRESS:
            if current:
                self._reserve(job, current.site_id, now, fallback_latency)
                return self._dispatch(job, current.site_id, now, "model_6_v4: immediate low slack")
            else:
                # Need to dispatch now but no immediate capacity. Can we schedule slightly in future?
                future_fallback = self._best_future(job, candidates, now, latest + timedelta(hours=1))
                if future_fallback:
                    self._reserve(job, future_fallback.site_id, future_fallback.timestamp, fallback_latency)
                    return self._dispatch(job, future_fallback.site_id, future_fallback.timestamp, "model_6_v4: fallback future")
                logger.debug("Job %s low slack, no current, no future_fallback. candidates=%d, immediate=%d",
                             job.job_id, len(candidates), len(immediate_candidates))
                return None

        future = self._best_future(job, candidates, now, latest)
        
        if future is None:
            if current:
                self._reserve(job, current.site_id, now, fallback_latency)
                return self._dispatch(job, current.site_id, now, "model_6_v4: immediate no future signal")
            else:
                logger.debug("Job %s no future, no current. candidates=%d, immediate=%d",
                             job.job_id, len(candidates), len(immediate_candidates))
                return None
                
        if current is None:
            self._reserve(job, future.site_id, future.timestamp, fallback_latency)
            return self._dispatch(
                job, future.site_id, future.timestamp, "model_6_v4: deferred (no immediate cap)"
            )

        carbon_gain = self._relative_gain(current.cfp_g, future.cfp_g)
        energy_gain = self._relative_gain(current.energy_wh, future.energy_wh)

        if self._worth_holding(carbon_gain, energy_gain, future.timestamp - now):
            self._reserve(job, future.site_id, future.timestamp, fallback_latency)
            return self._dispatch(
                job,
                future.site_id,
                future.timestamp,
                f"model_6_v4: deferred (carbon_gain={carbon_gain:.1%}, energy_gain={energy_gain:.1%})",
            )

        self._reserve(job, current.site_id, now, fallback_latency)
        return self._dispatch(
            job,
            current.site_id,
            now,
            f"model_6_v4: immediate best current",
        )

    def _best_future(
        self,
        job: Job,
        candidates: List[Site],
        now: datetime,
        latest: datetime,
    ) -> Optional[_Signal]:
        best: Optional[_Signal] = None
        best_score: Optional[tuple] = None
        candidate_ids = {site.site_id: site for site in candidates}

        cpu_minutes = job.cpu_minutes
        
        try:
            cores_required = max(1, int(float(job.metadata.get("ncores", job.metadata.get("cores", 1)))))
        except (TypeError, ValueError):
            cores_required = 1

        for site_id, site in candidate_ids.items():
            timeline = self._timeline.get(site_id, [])
            timestamps = self._timestamps.get(site_id, [])
            start_idx = bisect_right(timestamps, now)
            
            latency = timedelta(minutes=site.dispatch_latency)

            for signal in timeline[start_idx:]:
                if signal.timestamp > latest:
                    break

                if not self._can_reserve(job, site, signal.timestamp, cpu_minutes, latency, cores_required):
                    continue

                base_score = self._score_signal(signal)
                if best_score is None or base_score < best_score:
                    best = signal
                    best_score = base_score

        return best
        
    def _can_reserve(self, job: Job, site: Site, dispatch_ts: datetime, cpu_minutes: float, latency: timedelta, cores_required: int) -> bool:
        start_ts = dispatch_ts + latency
        # Subtract 1 microsecond so that perfectly aligned jobs don't reserve an extra bucket
        end_ts = start_ts + timedelta(minutes=cpu_minutes) - timedelta(microseconds=1)
            
        memory_required = job.memory_gb
        
        max_slots = int(site.capacity.max_concurrent_jobs * self.bucket_headroom)
        max_cores = int(site.capacity.max_cores * self.bucket_headroom)
        max_memory = float(site.capacity.max_memory_gb * self.bucket_headroom)
        
        curr = _bucket_start(start_ts)
        end_bucket = _bucket_start(end_ts)
        
        while curr <= end_bucket:
            res = self._reservations[site.site_id][curr]
            if res.get('slots', 0) + 1 > max_slots: return False
            if res.get('cores', 0) + cores_required > max_cores: return False
            if res.get('memory', 0.0) + memory_required > max_memory: return False
            curr += timedelta(minutes=15)
            
        return True

    def _reserve(self, job: Job, site_id: str, dispatch_ts: datetime, latency: timedelta) -> None:
        start_ts = dispatch_ts + latency
        end_ts = start_ts + timedelta(minutes=job.cpu_minutes) - timedelta(microseconds=1)
        
        try:
            cores_required = max(1, int(float(job.metadata.get("ncores", job.metadata.get("cores", 1)))))
        except (TypeError, ValueError):
            cores_required = 1
            
        memory_required = job.memory_gb
        
        curr = _bucket_start(start_ts)
        end_bucket = _bucket_start(end_ts)
        
        while curr <= end_bucket:
            res = self._reservations[site_id][curr]
            res['slots'] = res.get('slots', 0) + 1
            res['cores'] = res.get('cores', 0) + cores_required
            res['memory'] = res.get('memory', 0.0) + memory_required
            curr += timedelta(minutes=15)

    def _score_signal(self, signal: _Signal) -> tuple:
        if self.declared_objective == Objective.ENERGY:
            return (signal.energy_wh, signal.cfp_g, signal.timestamp, signal.site_id)
        if self.declared_objective == Objective.MAKESPAN:
            return (0.60 * signal.cfp_g + 0.40 * signal.energy_wh, signal.energy_wh, signal.timestamp, signal.site_id)
        return (0.80 * signal.cfp_g + 0.20 * signal.energy_wh, signal.energy_wh, signal.timestamp, signal.site_id)

    def _worth_holding(
        self,
        carbon_gain: float,
        energy_gain: float,
        hold: timedelta,
    ) -> bool:
        hold_hours = hold.total_seconds() / 3600.0

        if self.declared_objective == Objective.ENERGY:
            return energy_gain >= self.energy_saving_threshold and carbon_gain >= -0.05
        if self.declared_objective == Objective.MAKESPAN:
            return carbon_gain >= 0.20 and energy_gain >= 0.0 and hold_hours <= 2.0

        min_carbon_gain = self.carbon_saving_threshold + min(0.01, hold_hours * 0.001)
        return carbon_gain >= min_carbon_gain and energy_gain >= -0.10

    def _load_forecast_csv(self, raw_path: Optional[str]) -> None:
        if not raw_path:
            return
        path = Path(raw_path).expanduser()
        if not path.exists():
            return
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                for signal in self._iter_signals([row]):
                    self._signals.setdefault(signal.site_id, {})[signal.timestamp] = signal
        self._rebuild_timeline()

    def _rebuild_timeline(self) -> None:
        self._timeline = {
            site_id: sorted(by_ts.values(), key=lambda signal: signal.timestamp)
            for site_id, by_ts in self._signals.items()
        }
        self._timestamps = {
            site_id: [signal.timestamp for signal in timeline]
            for site_id, timeline in self._timeline.items()
        }
        if self._eval_end is None:
            all_latest = [timeline[-1].timestamp for timeline in self._timeline.values() if timeline]
            if all_latest:
                self._eval_end = min(all_latest)

    @staticmethod
    def _dispatch(job: Job, site_id: str, when: datetime, rationale: str) -> DispatchDecision:
        return DispatchDecision(
            job_id=job.job_id,
            site_id=site_id,
            dispatch_at=when,
            rationale=rationale,
        )

    @staticmethod
    def _has_capacity(site: Site, job: Job) -> bool:
        cores = job.metadata.get("ncores", job.metadata.get("cores", 1))
        try:
            cores_required = max(1, int(float(cores)))
        except (TypeError, ValueError):
            cores_required = 1
        return (
            site.capacity.available_slots > 0
            and site.capacity.available_cores >= cores_required
            and site.capacity.available_memory_gb >= job.memory_gb
        )

    @staticmethod
    def _relative_gain(current: float, future: float) -> float:
        if current <= 0 or not isfinite(current) or not isfinite(future):
            return 0.0
        return (current - future) / current

    @staticmethod
    def _iter_signals(*record_groups: Iterable[dict]) -> Iterable[_Signal]:
        for records in record_groups:
            for record in records:
                site_id = str(record.get("series_id", ""))
                ts_raw = str(record.get("forecast_timestamp_utc", ""))
                if not site_id or not ts_raw:
                    continue
                try:
                    timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    energy = float(record.get("energy_wh_pred", "nan"))
                    carbon = float(record.get("cfp_g_pred", "nan"))
                except (TypeError, ValueError):
                    continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestamp.astimezone(timezone.utc)
                if isfinite(energy) and isfinite(carbon):
                    yield _Signal(timestamp, site_id, max(0.0, energy), max(0.0, carbon))

    @staticmethod
    def _read_eval_end() -> Optional[datetime]:
        raw = os.environ.get("TASKB_EVAL_END", "").strip()
        if not raw:
            return None
        try:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _default_forecast_csv() -> Optional[str]:
        raw = os.environ.get("TASKB_FORECAST_CSV", "").strip()
        if raw:
            return raw
        candidate = Path(__file__).resolve().parents[3] / "task-a" / "model_6" / "outputs" / "forecast_submission.csv"
        return str(candidate) if candidate.exists() else None
