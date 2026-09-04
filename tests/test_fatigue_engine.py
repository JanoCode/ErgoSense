from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from ergosense.application.fatigue_engine import FatigueEngine
from ergosense.domain.fatigue import (
    FatigueEngineConfig,
    FatigueEvidence,
    FatigueLevel,
    FatigueState,
    FatigueTrend,
    clamp_score,
)
from ergosense.domain.longitudinal import (
    LongitudinalFeatures,
    MetricAggregation,
    MetricDeviation,
    MetricTrend,
    TimeWindow,
    TrendDirection,
)


def observed_at(seconds: int) -> datetime:
    return datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def make_aggregation(mean=None, minimum=None, maximum=None, std=None, count=0):
    return MetricAggregation(
        mean=mean,
        minimum=minimum,
        maximum=maximum,
        standard_deviation=std,
        sample_count=count,
    )


def make_deviation(
    current=None,
    baseline_mean=None,
    absolute_difference=None,
    z_score=None,
    sample_count=0,
    available=False,
):
    return MetricDeviation(
        current=current,
        baseline_mean=baseline_mean,
        absolute_difference=absolute_difference,
        z_score=z_score,
        baseline_sample_count=sample_count,
        baseline_available=available,
    )


def make_trend(direction=None, slope=None, count=0):
    return MetricTrend(direction=direction, slope=slope, sample_count=count)


def make_features(**overrides):
    values = {
        "session_id": uuid4(),
        "window": TimeWindow.one_minute(),
        "window_started_at": observed_at(0),
        "window_ended_at": observed_at(60),
        "elapsed_since_start": timedelta(minutes=20),
        "time_since_previous_observation": timedelta(seconds=5),
        "continuous_time_since_session_start": timedelta(minutes=20),
        "observation_count": 6,
        "ear_aggregation": make_aggregation(
            mean=0.28, minimum=0.27, maximum=0.29, std=0.01, count=6
        ),
        "perclos_aggregation": make_aggregation(
            mean=0.10, minimum=0.08, maximum=0.12, std=0.01, count=6
        ),
        "pitch_aggregation": make_aggregation(
            mean=3.0, minimum=2.0, maximum=4.0, std=0.5, count=6
        ),
        "yaw_aggregation": make_aggregation(
            mean=2.0, minimum=1.0, maximum=3.0, std=0.5, count=6
        ),
        "roll_aggregation": make_aggregation(
            mean=1.5, minimum=1.0, maximum=2.0, std=0.4, count=6
        ),
        "ear_deviation": make_deviation(
            current=0.28,
            baseline_mean=0.28,
            absolute_difference=0.0,
            z_score=0.0,
            sample_count=12,
            available=True,
        ),
        "perclos_deviation": make_deviation(
            current=0.10,
            baseline_mean=0.10,
            absolute_difference=0.0,
            z_score=0.0,
            sample_count=12,
            available=True,
        ),
        "pitch_deviation": make_deviation(
            current=3.0,
            baseline_mean=3.0,
            absolute_difference=0.0,
            z_score=0.0,
            sample_count=12,
            available=True,
        ),
        "yaw_deviation": make_deviation(
            current=2.0,
            baseline_mean=2.0,
            absolute_difference=0.0,
            z_score=0.0,
            sample_count=12,
            available=True,
        ),
        "roll_deviation": make_deviation(
            current=1.5,
            baseline_mean=1.5,
            absolute_difference=0.0,
            z_score=0.0,
            sample_count=12,
            available=True,
        ),
        "ear_trend": make_trend(direction=TrendDirection.STABLE, slope=0.0, count=6),
        "perclos_trend": make_trend(
            direction=TrendDirection.STABLE, slope=0.0, count=6
        ),
        "pitch_trend": make_trend(direction=TrendDirection.STABLE, slope=0.0, count=6),
        "yaw_trend": make_trend(direction=TrendDirection.STABLE, slope=0.0, count=6),
        "roll_trend": make_trend(direction=TrendDirection.STABLE, slope=0.0, count=6),
        "events": (),
    }
    values.update(overrides)
    return LongitudinalFeatures(**values)


def make_previous_state(score=40.0, trend=FatigueTrend.STABLE):
    component = FatigueEvidence(
        source="test",
        score=score,
        strength=score,
        persistence=score,
        explanation="test",
    )
    return FatigueState(
        session_id=uuid4(),
        calculated_at=observed_at(55),
        score=score,
        level=FatigueLevel.MODERATE,
        trend=trend,
        confidence=80.0,
        ocular_component=component,
        postural_component=component,
        temporal_component=component,
        convergence_component=component,
        reasons=("test",),
    )


def test_clamp_score_returns_valid_range_for_invalid_values():
    assert clamp_score(-5) == 0.0
    assert clamp_score(105) == 100.0
    assert clamp_score(float("nan")) == 0.0
    assert clamp_score(float("inf")) == 0.0


def test_level_thresholds_are_applied_to_scores():
    engine = FatigueEngine()

    assert engine.config.level_for_score(10) is FatigueLevel.NORMAL
    assert engine.config.level_for_score(20) is FatigueLevel.MILD
    assert engine.config.level_for_score(40) is FatigueLevel.MODERATE
    assert engine.config.level_for_score(60) is FatigueLevel.HIGH
    assert engine.config.level_for_score(80) is FatigueLevel.VERY_HIGH


def test_fatigue_state_clamps_score_and_confidence():
    component = FatigueEvidence(
        source="x",
        score=150,
        strength=150,
        persistence=-10,
        explanation="x",
    )
    state = FatigueState(
        session_id=None,
        calculated_at=observed_at(0),
        score=120,
        level=FatigueLevel.VERY_HIGH,
        trend=FatigueTrend.UNKNOWN,
        confidence=-5,
        ocular_component=component,
        postural_component=component,
        temporal_component=component,
        convergence_component=component,
    )

    assert state.score == 100.0
    assert state.confidence == 0.0
    assert state.ocular_component.score == 100.0
    assert state.ocular_component.persistence == 0.0


def test_engine_returns_normal_state_for_near_baseline_signals():
    engine = FatigueEngine(now=lambda: observed_at(60))

    state = engine.evaluate(make_features())

    assert 0 <= state.score <= 100
    assert state.level is FatigueLevel.NORMAL
    assert state.ocular_component.score < 25
    assert state.postural_component.score < 25
    assert state.confidence >= 80


def test_ocular_component_increases_with_high_perclos_and_low_ear():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        ear_deviation=make_deviation(
            current=0.22,
            baseline_mean=0.28,
            absolute_difference=-0.06,
            z_score=-2.4,
            sample_count=12,
            available=True,
        ),
        perclos_deviation=make_deviation(
            current=0.28,
            baseline_mean=0.10,
            absolute_difference=0.18,
            z_score=3.0,
            sample_count=12,
            available=True,
        ),
        ear_trend=make_trend(
            direction=TrendDirection.DECREASING, slope=-0.002, count=6
        ),
        perclos_trend=make_trend(
            direction=TrendDirection.INCREASING, slope=0.003, count=6
        ),
    )

    state = engine.evaluate(features)

    assert state.ocular_component.score > 70
    assert state.score > 30
    assert any("PERCLOS" in reason or "EAR" in reason for reason in state.reasons)


def test_ocular_component_remains_conservative_with_missing_data():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        observation_count=1,
        ear_aggregation=make_aggregation(count=0),
        perclos_aggregation=make_aggregation(count=0),
        ear_deviation=make_deviation(),
        perclos_deviation=make_deviation(),
        ear_trend=make_trend(),
        perclos_trend=make_trend(),
    )

    state = engine.evaluate(features)

    assert state.ocular_component.score <= 10
    assert state.confidence < 70


def test_postural_component_stays_low_near_baseline():
    engine = FatigueEngine(now=lambda: observed_at(60))
    state = engine.evaluate(make_features())

    assert state.postural_component.score < 20


def test_postural_component_increases_with_sustained_deviation():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        pitch_deviation=make_deviation(
            current=14.0,
            baseline_mean=3.0,
            absolute_difference=11.0,
            z_score=2.8,
            sample_count=12,
            available=True,
        ),
        yaw_deviation=make_deviation(
            current=9.0,
            baseline_mean=2.0,
            absolute_difference=7.0,
            z_score=1.8,
            sample_count=12,
            available=True,
        ),
        pitch_trend=make_trend(
            direction=TrendDirection.INCREASING, slope=0.12, count=6
        ),
        yaw_trend=make_trend(direction=TrendDirection.INCREASING, slope=0.08, count=6),
        pitch_aggregation=make_aggregation(
            mean=12.0, minimum=8.0, maximum=14.0, std=2.5, count=6
        ),
        yaw_aggregation=make_aggregation(
            mean=8.0, minimum=6.0, maximum=9.0, std=1.5, count=6
        ),
    )

    state = engine.evaluate(features)

    assert state.postural_component.score > 40
    assert any(
        "postural" in reason.lower()
        or "postural" in state.postural_component.explanation.lower()
        for reason in state.reasons + (state.postural_component.explanation,)
    )


def test_postural_component_handles_missing_baseline_and_zero_std_conservatively():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        pitch_deviation=make_deviation(
            current=14.0,
            baseline_mean=None,
            absolute_difference=None,
            z_score=None,
            sample_count=0,
            available=False,
        ),
        pitch_aggregation=make_aggregation(
            mean=14.0, minimum=14.0, maximum=14.0, std=0.0, count=1
        ),
    )

    state = engine.evaluate(features)

    assert state.postural_component.score < 25


def test_temporal_component_is_low_for_short_session_and_higher_for_longer_session():
    engine = FatigueEngine(now=lambda: observed_at(60))
    short_state = engine.evaluate(
        make_features(continuous_time_since_session_start=timedelta(minutes=10))
    )
    long_state = engine.evaluate(
        make_features(continuous_time_since_session_start=timedelta(hours=3))
    )

    assert short_state.temporal_component.score == 0.0
    assert long_state.temporal_component.score > short_state.temporal_component.score


def test_temporal_component_handles_missing_time():
    engine = FatigueEngine(now=lambda: observed_at(60))
    state = engine.evaluate(make_features(continuous_time_since_session_start=None))

    assert state.temporal_component.score == 0.0
    assert state.temporal_component.explanation.startswith("Sin suficiente")


def test_convergence_component_rewards_multiple_independent_signals_but_stays_bounded():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        continuous_time_since_session_start=timedelta(hours=3),
        ear_deviation=make_deviation(
            current=0.22,
            baseline_mean=0.28,
            absolute_difference=-0.06,
            z_score=-2.4,
            sample_count=12,
            available=True,
        ),
        perclos_deviation=make_deviation(
            current=0.30,
            baseline_mean=0.10,
            absolute_difference=0.20,
            z_score=4.0,
            sample_count=12,
            available=True,
        ),
        ear_trend=make_trend(
            direction=TrendDirection.DECREASING, slope=-0.002, count=6
        ),
        perclos_trend=make_trend(
            direction=TrendDirection.INCREASING, slope=0.004, count=6
        ),
        pitch_deviation=make_deviation(
            current=15.0,
            baseline_mean=3.0,
            absolute_difference=12.0,
            z_score=3.0,
            sample_count=12,
            available=True,
        ),
        pitch_trend=make_trend(
            direction=TrendDirection.INCREASING, slope=0.10, count=6
        ),
        pitch_aggregation=make_aggregation(
            mean=13.0, minimum=8.0, maximum=15.0, std=3.0, count=6
        ),
    )

    state = engine.evaluate(features)

    assert state.convergence_component.score <= 100
    assert state.convergence_component.score >= 45


def test_confidence_drops_with_few_observations_missing_baseline_and_missing_metrics():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        observation_count=1,
        continuous_time_since_session_start=timedelta(minutes=1),
        ear_deviation=make_deviation(
            current=None,
            baseline_mean=None,
            absolute_difference=None,
            z_score=None,
            sample_count=0,
            available=False,
        ),
        perclos_deviation=make_deviation(
            current=None,
            baseline_mean=None,
            absolute_difference=None,
            z_score=None,
            sample_count=0,
            available=False,
        ),
        pitch_deviation=make_deviation(
            current=None,
            baseline_mean=None,
            absolute_difference=None,
            z_score=None,
            sample_count=0,
            available=False,
        ),
        yaw_deviation=make_deviation(
            current=None,
            baseline_mean=None,
            absolute_difference=None,
            z_score=None,
            sample_count=0,
            available=False,
        ),
        roll_deviation=make_deviation(
            current=None,
            baseline_mean=None,
            absolute_difference=None,
            z_score=None,
            sample_count=0,
            available=False,
        ),
    )

    state = engine.evaluate(features)

    assert state.confidence < 30


def test_stabilization_limits_large_upward_jump_from_previous_state():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        continuous_time_since_session_start=timedelta(hours=3),
        ear_deviation=make_deviation(
            current=0.20,
            baseline_mean=0.28,
            absolute_difference=-0.08,
            z_score=-3.0,
            sample_count=12,
            available=True,
        ),
        perclos_deviation=make_deviation(
            current=0.30,
            baseline_mean=0.10,
            absolute_difference=0.20,
            z_score=4.0,
            sample_count=12,
            available=True,
        ),
        ear_trend=make_trend(
            direction=TrendDirection.DECREASING, slope=-0.003, count=6
        ),
        perclos_trend=make_trend(
            direction=TrendDirection.INCREASING, slope=0.004, count=6
        ),
    )

    state = engine.evaluate(features, previous_state=make_previous_state(score=10.0))

    assert state.score <= 28.0


def test_stabilization_reduces_small_change_smoothly():
    engine = FatigueEngine(now=lambda: observed_at(60))
    previous = make_previous_state(score=40.0)
    state = engine.evaluate(make_features(), previous_state=previous)

    assert 20.0 <= state.score <= 40.0


def test_trend_uses_previous_state_delta_when_available():
    engine = FatigueEngine(now=lambda: observed_at(60))
    strong_features = make_features(
        continuous_time_since_session_start=timedelta(hours=3),
        ear_deviation=make_deviation(
            current=0.20,
            baseline_mean=0.28,
            absolute_difference=-0.08,
            z_score=-3.0,
            sample_count=12,
            available=True,
        ),
        perclos_deviation=make_deviation(
            current=0.30,
            baseline_mean=0.10,
            absolute_difference=0.20,
            z_score=4.0,
            sample_count=12,
            available=True,
        ),
        ear_trend=make_trend(
            direction=TrendDirection.DECREASING, slope=-0.003, count=6
        ),
        perclos_trend=make_trend(
            direction=TrendDirection.INCREASING, slope=0.004, count=6
        ),
    )
    up = engine.evaluate(
        strong_features, previous_state=make_previous_state(score=10.0)
    )
    down = engine.evaluate(
        make_features(), previous_state=make_previous_state(score=80.0)
    )
    stable = engine.evaluate(
        make_features(), previous_state=make_previous_state(score=18.0)
    )

    assert up.trend is FatigueTrend.INCREASING
    assert down.trend is FatigueTrend.DECREASING
    assert stable.trend is FatigueTrend.STABLE


def test_trend_can_be_unknown_without_previous_state_and_without_feature_trends():
    engine = FatigueEngine(now=lambda: observed_at(60))
    features = make_features(
        ear_trend=make_trend(),
        perclos_trend=make_trend(),
        pitch_trend=make_trend(),
        yaw_trend=make_trend(),
        roll_trend=make_trend(),
    )

    state = engine.evaluate(features)

    assert state.trend is FatigueTrend.UNKNOWN


def test_integration_from_longitudinal_features_to_fatigue_state():
    engine = FatigueEngine(now=lambda: observed_at(60))
    state = engine.evaluate(make_features())

    assert isinstance(state, FatigueState)
    assert state.session_id is not None
    assert isinstance(state.level, FatigueLevel)
    assert isinstance(state.trend, FatigueTrend)
    assert 0 <= state.score <= 100
    assert 0 <= state.confidence <= 100


def test_engine_uses_configurable_weights_and_thresholds():
    config = FatigueEngineConfig(
        ocular_weight=0.50,
        postural_weight=0.20,
        temporal_weight=0.10,
        convergence_weight=0.20,
    )
    engine = FatigueEngine(config=config, now=lambda: observed_at(60))
    state = engine.evaluate(make_features())

    assert state.level is config.level_for_score(state.score)
