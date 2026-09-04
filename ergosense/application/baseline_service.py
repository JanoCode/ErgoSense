import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from ergosense.domain.baseline import (
    BaselineConfidence,
    EyeBaseline,
    HeadBaseline,
    MetricBaseline,
    PersonalBaseline,
)
from ergosense.domain.monitoring import FrameObservation
from ergosense.domain.session import MonitoringSession

DEFAULT_BASELINE_CALIBRATION_DURATION = timedelta(minutes=2)
DEFAULT_BASELINE_MIN_SAMPLES = 5


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass
class RunningMetricStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def snapshot(self) -> MetricBaseline:
        if self.count <= 1:
            standard_deviation = 0.0
        else:
            standard_deviation = math.sqrt(self.m2 / (self.count - 1))
        return MetricBaseline(
            mean=self.mean,
            standard_deviation=standard_deviation,
            sample_count=self.count,
        )


class BaselineService:
    """Build an in-memory personal baseline from valid observations."""

    def __init__(
        self,
        *,
        now: Callable[[], datetime] = utc_now,
        calibration_duration: timedelta = DEFAULT_BASELINE_CALIBRATION_DURATION,
        min_samples_per_metric: int = DEFAULT_BASELINE_MIN_SAMPLES,
        observation_filter: Callable[[FrameObservation], bool] | None = None,
    ):
        if calibration_duration <= timedelta(0):
            raise ValueError("calibration_duration must be positive")
        if min_samples_per_metric <= 0:
            raise ValueError("min_samples_per_metric must be positive")
        self._now = now
        self.calibration_duration = calibration_duration
        self.min_samples_per_metric = min_samples_per_metric
        self.observation_filter = observation_filter or self._default_observation_filter
        self._session_id = None
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._ear = RunningMetricStats()
        self._perclos = RunningMetricStats()
        self._pitch = RunningMetricStats()
        self._yaw = RunningMetricStats()
        self._roll = RunningMetricStats()

    def start_calibration(
        self,
        *,
        session: MonitoringSession | None = None,
        started_at: datetime | None = None,
    ) -> PersonalBaseline:
        self._session_id = session.session_id if session is not None else None
        self._started_at = (
            session.started_at if session is not None else (started_at or self._now())
        )
        _require_aware_datetime(self._started_at, "started_at")
        self._completed_at = None
        self._ear = RunningMetricStats()
        self._perclos = RunningMetricStats()
        self._pitch = RunningMetricStats()
        self._yaw = RunningMetricStats()
        self._roll = RunningMetricStats()
        return self.get_baseline()

    def add_observation(self, observation: FrameObservation) -> PersonalBaseline:
        if self._started_at is None:
            raise ValueError("baseline calibration has not been started")
        if self._completed_at is not None:
            raise ValueError("baseline calibration has already been finished")
        if observation.observed_at < self._started_at:
            raise ValueError("observation cannot be earlier than calibration start")
        if not self.observation_filter(observation):
            return self.get_baseline(at=observation.observed_at)

        self._add_metric(self._ear, observation.ear)
        if observation.perclos_ready:
            self._add_metric(self._perclos, observation.perclos)
        self._add_metric(self._pitch, observation.pitch)
        self._add_metric(self._yaw, observation.yaw)
        self._add_metric(self._roll, observation.roll)
        return self.get_baseline(at=observation.observed_at)

    def finish_calibration(
        self, *, completed_at: datetime | None = None
    ) -> PersonalBaseline:
        if self._started_at is None:
            raise ValueError("baseline calibration has not been started")
        finished = completed_at or self._now()
        _require_aware_datetime(finished, "completed_at")
        if finished < self._started_at:
            raise ValueError("calibration cannot finish before it starts")
        self._completed_at = finished
        return self.get_baseline(at=finished)

    def is_ready(self, *, at: datetime | None = None) -> bool:
        return self.get_baseline(at=at).confidence is BaselineConfidence.READY

    def get_baseline(self, *, at: datetime | None = None) -> PersonalBaseline:
        return PersonalBaseline(
            session_id=self._session_id,
            started_at=self._started_at,
            completed_at=self._completed_at,
            calibration_duration=self.calibration_duration,
            confidence=self._get_confidence(at=at),
            eye=EyeBaseline(
                ear=self._ear.snapshot(),
                perclos=self._perclos.snapshot(),
            ),
            head=HeadBaseline(
                pitch=self._pitch.snapshot(),
                yaw=self._yaw.snapshot(),
                roll=self._roll.snapshot(),
            ),
        )

    def _get_confidence(self, *, at: datetime | None = None) -> BaselineConfidence:
        if self._started_at is None:
            return BaselineConfidence.INSUFFICIENT_DATA
        current_at = self._completed_at or at or self._now()
        elapsed = current_at - self._started_at
        if self._has_sufficient_samples() and elapsed >= self.calibration_duration:
            return BaselineConfidence.READY
        if self._has_any_samples():
            return BaselineConfidence.CALIBRATING
        return BaselineConfidence.INSUFFICIENT_DATA

    def _has_any_samples(self) -> bool:
        return any(
            metric.count > 0
            for metric in (self._ear, self._perclos, self._pitch, self._yaw, self._roll)
        )

    def _has_sufficient_samples(self) -> bool:
        return all(
            metric.count >= self.min_samples_per_metric
            for metric in (self._ear, self._perclos, self._pitch, self._yaw, self._roll)
        )

    @staticmethod
    def _default_observation_filter(observation: FrameObservation) -> bool:
        return observation.face_detected

    @staticmethod
    def _add_metric(stats: RunningMetricStats, value: float | None) -> None:
        if value is None:
            return
        if not math.isfinite(value):
            return
        stats.add(float(value))
