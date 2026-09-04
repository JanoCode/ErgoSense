from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass(frozen=True)
class TimeWindow:
    name: str
    duration: timedelta

    ONE_MINUTE_DURATION = timedelta(minutes=1)
    FIVE_MINUTES_DURATION = timedelta(minutes=5)
    FIFTEEN_MINUTES_DURATION = timedelta(minutes=15)
    THIRTY_MINUTES_DURATION = timedelta(minutes=30)

    def __post_init__(self) -> None:
        if self.duration <= timedelta(0):
            raise ValueError("window duration must be positive")

    @classmethod
    def one_minute(cls) -> "TimeWindow":
        return cls(name="1m", duration=cls.ONE_MINUTE_DURATION)

    @classmethod
    def five_minutes(cls) -> "TimeWindow":
        return cls(name="5m", duration=cls.FIVE_MINUTES_DURATION)

    @classmethod
    def fifteen_minutes(cls) -> "TimeWindow":
        return cls(name="15m", duration=cls.FIFTEEN_MINUTES_DURATION)

    @classmethod
    def thirty_minutes(cls) -> "TimeWindow":
        return cls(name="30m", duration=cls.THIRTY_MINUTES_DURATION)


@dataclass(frozen=True)
class MetricAggregation:
    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    standard_deviation: float | None = None
    sample_count: int = 0


@dataclass(frozen=True)
class MetricDeviation:
    current: float | None = None
    baseline_mean: float | None = None
    absolute_difference: float | None = None
    z_score: float | None = None
    baseline_sample_count: int = 0
    baseline_available: bool = False


@dataclass(frozen=True)
class MetricTrend:
    slope: float | None = None
    direction: TrendDirection | None = None
    sample_count: int = 0


@dataclass(frozen=True)
class ProlongedEvent:
    event_type: str
    started_at: datetime
    ended_at: datetime
    duration: timedelta

    def __post_init__(self) -> None:
        _require_aware_datetime(self.started_at, "started_at")
        _require_aware_datetime(self.ended_at, "ended_at")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be earlier than started_at")
        if self.duration < timedelta(0):
            raise ValueError("duration cannot be negative")


@dataclass(frozen=True)
class LongitudinalFeatures:
    session_id: UUID | None
    window: TimeWindow
    window_started_at: datetime | None
    window_ended_at: datetime | None
    elapsed_since_start: timedelta | None
    time_since_previous_observation: timedelta | None
    continuous_time_since_session_start: timedelta | None
    observation_count: int
    ear_aggregation: MetricAggregation
    perclos_aggregation: MetricAggregation
    pitch_aggregation: MetricAggregation
    yaw_aggregation: MetricAggregation
    roll_aggregation: MetricAggregation
    ear_deviation: MetricDeviation
    perclos_deviation: MetricDeviation
    pitch_deviation: MetricDeviation
    yaw_deviation: MetricDeviation
    roll_deviation: MetricDeviation
    ear_trend: MetricTrend
    perclos_trend: MetricTrend
    pitch_trend: MetricTrend
    yaw_trend: MetricTrend
    roll_trend: MetricTrend
    events: tuple[ProlongedEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.window_started_at is not None:
            _require_aware_datetime(self.window_started_at, "window_started_at")
        if self.window_ended_at is not None:
            _require_aware_datetime(self.window_ended_at, "window_ended_at")
        if (
            self.window_started_at is not None
            and self.window_ended_at is not None
            and self.window_ended_at < self.window_started_at
        ):
            raise ValueError("window_ended_at cannot be earlier than window_started_at")
