from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from ergosense.application.longitudinal_feature_service import (
    LongitudinalFeatureService,
)
from ergosense.domain.baseline import (
    BaselineConfidence,
    EyeBaseline,
    HeadBaseline,
    MetricBaseline,
    PersonalBaseline,
)
from ergosense.domain.longitudinal import TimeWindow, TrendDirection
from ergosense.domain.monitoring import AnalysisResult, AnalysisState, FrameObservation
from ergosense.domain.session import SessionObservation


def observed_at(seconds: int) -> datetime:
    return datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def make_session_observation(seconds: int, **metrics) -> SessionObservation:
    observation = FrameObservation(
        timestamp=float(seconds),
        observed_at=observed_at(seconds),
        face_detected=True,
        ear=metrics.get("ear", 0.2),
        perclos=metrics.get("perclos", 0.1),
        perclos_ready=metrics.get("perclos_ready", True),
        pitch=metrics.get("pitch", 1.0),
        yaw=metrics.get("yaw", 2.0),
        roll=metrics.get("roll", 3.0),
    )
    return SessionObservation(
        session_id=uuid4(),
        captured_at=observed_at(seconds),
        elapsed_since_start=timedelta(seconds=seconds),
        result=AnalysisResult(observation=observation, state=AnalysisState()),
    )


def make_baseline(
    *,
    ear=(0.2, 0.05, 10),
    perclos=(0.1, 0.02, 10),
    pitch=(1.0, 0.5, 10),
    yaw=(2.0, 0.5, 10),
    roll=(3.0, 0.5, 10),
) -> PersonalBaseline:
    return PersonalBaseline(
        confidence=BaselineConfidence.READY,
        eye=EyeBaseline(
            ear=MetricBaseline(*ear),
            perclos=MetricBaseline(*perclos),
        ),
        head=HeadBaseline(
            pitch=MetricBaseline(*pitch),
            yaw=MetricBaseline(*yaw),
            roll=MetricBaseline(*roll),
        ),
    )


def test_deviation_with_valid_baseline_includes_absolute_difference_and_z_score():
    service = LongitudinalFeatureService()
    features = service.extract_features(
        [make_session_observation(60, ear=0.30)],
        make_baseline(ear=(0.2, 0.05, 10)),
        TimeWindow.one_minute(),
    )

    assert features.ear_deviation.current == pytest.approx(0.30)
    assert features.ear_deviation.baseline_mean == pytest.approx(0.2)
    assert features.ear_deviation.absolute_difference == pytest.approx(0.10)
    assert features.ear_deviation.z_score == pytest.approx(2.0)
    assert features.ear_deviation.baseline_sample_count == 10
    assert features.ear_deviation.baseline_available


def test_deviation_marks_baseline_unavailable_when_metric_has_no_samples():
    service = LongitudinalFeatureService()
    baseline = make_baseline(ear=(0.0, 0.0, 0))

    features = service.extract_features(
        [make_session_observation(60, ear=0.30)], baseline, TimeWindow.one_minute()
    )

    assert features.ear_deviation.baseline_available is False
    assert features.ear_deviation.baseline_mean is None
    assert features.ear_deviation.absolute_difference is None
    assert features.ear_deviation.z_score is None


def test_deviation_skips_z_score_when_baseline_standard_deviation_is_zero():
    service = LongitudinalFeatureService()
    baseline = make_baseline(ear=(0.25, 0.0, 10))

    features = service.extract_features(
        [make_session_observation(60, ear=0.30)], baseline, TimeWindow.one_minute()
    )

    assert features.ear_deviation.absolute_difference == pytest.approx(0.05)
    assert features.ear_deviation.z_score is None


def test_aggregation_computes_mean_min_max_std_and_sample_count():
    service = LongitudinalFeatureService()
    observations = [
        make_session_observation(10, ear=0.10),
        make_session_observation(20, ear=0.20),
        make_session_observation(30, ear=0.30),
    ]

    features = service.extract_features(
        observations, make_baseline(), TimeWindow.one_minute()
    )

    assert features.ear_aggregation.mean == pytest.approx(0.20)
    assert features.ear_aggregation.minimum == pytest.approx(0.10)
    assert features.ear_aggregation.maximum == pytest.approx(0.30)
    assert features.ear_aggregation.standard_deviation == pytest.approx(0.10)
    assert features.ear_aggregation.sample_count == 3


def test_window_filters_out_observations_before_start():
    service = LongitudinalFeatureService()
    observations = [
        make_session_observation(0, ear=0.10),
        make_session_observation(30, ear=0.20),
        make_session_observation(60, ear=0.30),
    ]

    features = service.extract_features(
        observations,
        make_baseline(),
        TimeWindow.one_minute(),
        at=observed_at(70),
    )

    assert features.observation_count == 2
    assert features.ear_aggregation.sample_count == 2
    assert features.ear_aggregation.minimum == pytest.approx(0.20)


def test_window_can_be_empty_even_with_observations_outside_range():
    service = LongitudinalFeatureService()
    features = service.extract_features(
        [make_session_observation(0, ear=0.10)],
        make_baseline(),
        TimeWindow.one_minute(),
        at=observed_at(120),
    )

    assert features.observation_count == 0
    assert features.window_started_at == observed_at(60)
    assert features.window_ended_at == observed_at(120)
    assert features.ear_aggregation.sample_count == 0
    assert features.ear_deviation.current is None


def test_empty_input_returns_empty_features_without_window_bounds_when_at_is_missing():
    service = LongitudinalFeatureService()
    features = service.extract_features([], make_baseline(), TimeWindow.one_minute())

    assert features.observation_count == 0
    assert features.window_started_at is None
    assert features.window_ended_at is None


def test_trend_can_be_increasing():
    service = LongitudinalFeatureService()
    observations = [
        make_session_observation(10, ear=0.10),
        make_session_observation(20, ear=0.20),
        make_session_observation(30, ear=0.30),
    ]

    features = service.extract_features(
        observations, make_baseline(), TimeWindow.one_minute()
    )

    assert features.ear_trend.direction is TrendDirection.INCREASING
    assert features.ear_trend.slope == pytest.approx(0.01)


def test_trend_can_be_decreasing():
    service = LongitudinalFeatureService()
    observations = [
        make_session_observation(10, pitch=3.0),
        make_session_observation(20, pitch=2.0),
        make_session_observation(30, pitch=1.0),
    ]

    features = service.extract_features(
        observations, make_baseline(), TimeWindow.one_minute()
    )

    assert features.pitch_trend.direction is TrendDirection.DECREASING
    assert features.pitch_trend.slope == pytest.approx(-0.1)


def test_trend_can_be_stable_with_small_variation_inside_epsilon():
    service = LongitudinalFeatureService(trend_epsilon=0.01)
    observations = [
        make_session_observation(10, roll=1.000),
        make_session_observation(20, roll=1.002),
        make_session_observation(30, roll=1.004),
    ]

    features = service.extract_features(
        observations, make_baseline(), TimeWindow.one_minute()
    )

    assert features.roll_trend.direction is TrendDirection.STABLE


def test_trend_is_unavailable_with_fewer_than_two_valid_samples():
    service = LongitudinalFeatureService()
    features = service.extract_features(
        [make_session_observation(10, yaw=None)],
        make_baseline(),
        TimeWindow.one_minute(),
    )

    assert features.yaw_trend.direction is None
    assert features.yaw_trend.slope is None
    assert features.yaw_trend.sample_count == 0


def test_none_nan_and_infinite_values_are_ignored():
    service = LongitudinalFeatureService()
    observations = [
        make_session_observation(10, ear=None),
        make_session_observation(20, ear=float("nan")),
        make_session_observation(30, ear=float("inf")),
        make_session_observation(40, ear=0.25),
    ]

    features = service.extract_features(
        observations, make_baseline(), TimeWindow.one_minute()
    )

    assert features.ear_aggregation.sample_count == 1
    assert features.ear_aggregation.mean == pytest.approx(0.25)
    assert features.ear_deviation.current == pytest.approx(0.25)


def test_elapsed_since_start_and_previous_observation_time_are_exposed():
    service = LongitudinalFeatureService()
    observations = [
        make_session_observation(10),
        make_session_observation(25),
        make_session_observation(40),
    ]

    features = service.extract_features(
        observations, make_baseline(), TimeWindow.one_minute()
    )

    assert features.elapsed_since_start == timedelta(seconds=40)
    assert features.time_since_previous_observation == timedelta(seconds=15)
    assert features.continuous_time_since_session_start == timedelta(seconds=40)
    assert features.window.duration == timedelta(minutes=1)


def test_presets_expose_supported_window_durations():
    assert TimeWindow.one_minute().duration == timedelta(minutes=1)
    assert TimeWindow.five_minutes().duration == timedelta(minutes=5)
    assert TimeWindow.fifteen_minutes().duration == timedelta(minutes=15)
    assert TimeWindow.thirty_minutes().duration == timedelta(minutes=30)
