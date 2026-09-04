import math
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

from ergosense.domain.baseline import MetricBaseline, PersonalBaseline
from ergosense.domain.longitudinal import (
    LongitudinalFeatures,
    MetricAggregation,
    MetricDeviation,
    MetricTrend,
    TimeWindow,
    TrendDirection,
)
from ergosense.domain.session import SessionObservation

DEFAULT_TREND_EPSILON = 1e-4


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class LongitudinalFeatureService:
    """Extract temporal features from session observations and a baseline."""

    def __init__(self, *, trend_epsilon: float = DEFAULT_TREND_EPSILON):
        if trend_epsilon < 0:
            raise ValueError("trend_epsilon cannot be negative")
        self.trend_epsilon = trend_epsilon

    def extract_features(
        self,
        observations: Iterable[SessionObservation],
        baseline: PersonalBaseline,
        window: TimeWindow,
        at: datetime | None = None,
    ) -> LongitudinalFeatures:
        ordered_observations = tuple(
            sorted(observations, key=lambda item: item.captured_at)
        )
        if at is not None:
            _require_aware_datetime(at, "at")

        window_ended_at = at or self._resolve_window_end(ordered_observations)
        if window_ended_at is None:
            return self._empty_features(window)

        window_started_at = window_ended_at - window.duration
        window_observations = tuple(
            observation
            for observation in ordered_observations
            if window_started_at <= observation.captured_at <= window_ended_at
        )
        latest_observation = window_observations[-1] if window_observations else None
        previous_observation = (
            window_observations[-2] if len(window_observations) >= 2 else None
        )

        return LongitudinalFeatures(
            session_id=(latest_observation.session_id if latest_observation else None),
            window=window,
            window_started_at=window_started_at,
            window_ended_at=window_ended_at,
            elapsed_since_start=(
                latest_observation.elapsed_since_start if latest_observation else None
            ),
            time_since_previous_observation=(
                latest_observation.captured_at - previous_observation.captured_at
                if latest_observation is not None and previous_observation is not None
                else None
            ),
            continuous_time_since_session_start=(
                latest_observation.elapsed_since_start if latest_observation else None
            ),
            observation_count=len(window_observations),
            ear_aggregation=self._aggregate_metric(window_observations, "ear"),
            perclos_aggregation=self._aggregate_metric(window_observations, "perclos"),
            pitch_aggregation=self._aggregate_metric(window_observations, "pitch"),
            yaw_aggregation=self._aggregate_metric(window_observations, "yaw"),
            roll_aggregation=self._aggregate_metric(window_observations, "roll"),
            ear_deviation=self._compute_deviation(
                window_observations, "ear", baseline.eye.ear
            ),
            perclos_deviation=self._compute_deviation(
                window_observations, "perclos", baseline.eye.perclos
            ),
            pitch_deviation=self._compute_deviation(
                window_observations, "pitch", baseline.head.pitch
            ),
            yaw_deviation=self._compute_deviation(
                window_observations, "yaw", baseline.head.yaw
            ),
            roll_deviation=self._compute_deviation(
                window_observations, "roll", baseline.head.roll
            ),
            ear_trend=self._compute_trend(window_observations, "ear"),
            perclos_trend=self._compute_trend(window_observations, "perclos"),
            pitch_trend=self._compute_trend(window_observations, "pitch"),
            yaw_trend=self._compute_trend(window_observations, "yaw"),
            roll_trend=self._compute_trend(window_observations, "roll"),
            events=(),
        )

    @staticmethod
    def _resolve_window_end(
        observations: Sequence[SessionObservation],
    ) -> datetime | None:
        if not observations:
            return None
        return observations[-1].captured_at

    @staticmethod
    def _empty_features(window: TimeWindow) -> LongitudinalFeatures:
        empty_aggregation = MetricAggregation()
        empty_deviation = MetricDeviation()
        empty_trend = MetricTrend()
        return LongitudinalFeatures(
            session_id=None,
            window=window,
            window_started_at=None,
            window_ended_at=None,
            elapsed_since_start=None,
            time_since_previous_observation=None,
            continuous_time_since_session_start=None,
            observation_count=0,
            ear_aggregation=empty_aggregation,
            perclos_aggregation=empty_aggregation,
            pitch_aggregation=empty_aggregation,
            yaw_aggregation=empty_aggregation,
            roll_aggregation=empty_aggregation,
            ear_deviation=empty_deviation,
            perclos_deviation=empty_deviation,
            pitch_deviation=empty_deviation,
            yaw_deviation=empty_deviation,
            roll_deviation=empty_deviation,
            ear_trend=empty_trend,
            perclos_trend=empty_trend,
            pitch_trend=empty_trend,
            yaw_trend=empty_trend,
            roll_trend=empty_trend,
            events=(),
        )

    def _aggregate_metric(
        self, observations: Sequence[SessionObservation], metric_name: str
    ) -> MetricAggregation:
        values = [
            value
            for observation in observations
            if (value := self._get_metric_value(observation, metric_name)) is not None
        ]
        if not values:
            return MetricAggregation(sample_count=0)
        mean = sum(values) / len(values)
        minimum = min(values)
        maximum = max(values)
        if len(values) <= 1:
            standard_deviation = 0.0
        else:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            standard_deviation = math.sqrt(variance)
        return MetricAggregation(
            mean=mean,
            minimum=minimum,
            maximum=maximum,
            standard_deviation=standard_deviation,
            sample_count=len(values),
        )

    def _compute_deviation(
        self,
        observations: Sequence[SessionObservation],
        metric_name: str,
        baseline_metric: MetricBaseline,
    ) -> MetricDeviation:
        current = self._get_latest_metric_value(observations, metric_name)
        baseline_available = baseline_metric.sample_count > 0 and math.isfinite(
            baseline_metric.mean
        )
        baseline_mean = baseline_metric.mean if baseline_available else None
        absolute_difference = None
        if current is not None and baseline_mean is not None:
            difference = current - baseline_mean
            if math.isfinite(difference):
                absolute_difference = difference
        z_score = None
        if (
            current is not None
            and baseline_available
            and baseline_metric.sample_count > 1
            and baseline_metric.standard_deviation > 0
            and math.isfinite(baseline_metric.standard_deviation)
        ):
            score = (
                current - baseline_metric.mean
            ) / baseline_metric.standard_deviation
            if math.isfinite(score):
                z_score = score
        return MetricDeviation(
            current=current,
            baseline_mean=baseline_mean,
            absolute_difference=absolute_difference,
            z_score=z_score,
            baseline_sample_count=baseline_metric.sample_count,
            baseline_available=baseline_available,
        )

    def _compute_trend(
        self, observations: Sequence[SessionObservation], metric_name: str
    ) -> MetricTrend:
        points = [
            (observation.captured_at, value)
            for observation in observations
            if (value := self._get_metric_value(observation, metric_name)) is not None
        ]
        if len(points) < 2:
            return MetricTrend(sample_count=len(points))

        first_time = points[0][0]
        x_values = [
            (captured_at - first_time).total_seconds() for captured_at, _ in points
        ]
        y_values = [value for _, value in points]

        mean_x = sum(x_values) / len(x_values)
        mean_y = sum(y_values) / len(y_values)
        denominator = sum((x - mean_x) ** 2 for x in x_values)
        if denominator <= 0:
            return MetricTrend(sample_count=len(points))

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        slope = numerator / denominator
        if not math.isfinite(slope):
            return MetricTrend(sample_count=len(points))

        if slope > self.trend_epsilon:
            direction = TrendDirection.INCREASING
        elif slope < -self.trend_epsilon:
            direction = TrendDirection.DECREASING
        else:
            direction = TrendDirection.STABLE
        return MetricTrend(slope=slope, direction=direction, sample_count=len(points))

    @staticmethod
    def _get_latest_metric_value(
        observations: Sequence[SessionObservation], metric_name: str
    ) -> float | None:
        for observation in reversed(observations):
            value = LongitudinalFeatureService._get_metric_value(
                observation, metric_name
            )
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_metric_value(
        observation: SessionObservation, metric_name: str
    ) -> float | None:
        value = getattr(observation.result.observation, metric_name)
        if value is None or not math.isfinite(value):
            return None
        return float(value)


def extract_features(
    observations: Iterable[SessionObservation],
    baseline: PersonalBaseline,
    window: TimeWindow,
    at: datetime | None = None,
) -> LongitudinalFeatures:
    return LongitudinalFeatureService().extract_features(
        observations=observations,
        baseline=baseline,
        window=window,
        at=at,
    )
