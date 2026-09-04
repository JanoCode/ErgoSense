from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from ergosense.application.fatigue_assessment_service import (
    FatigueAssessmentConfig,
    FatigueAssessmentService,
)
from ergosense.application.fatigue_engine import FatigueEngine
from ergosense.application.longitudinal_feature_service import (
    LongitudinalFeatureService,
)
from ergosense.domain.assessment import AssessmentTrend, WindowAssessmentStatus
from ergosense.domain.baseline import (
    BaselineConfidence,
    EyeBaseline,
    HeadBaseline,
    MetricBaseline,
    PersonalBaseline,
)
from ergosense.domain.fatigue import (
    FatigueEvidence,
    FatigueLevel,
    FatigueState,
    FatigueTrend,
)
from ergosense.domain.longitudinal import (
    LongitudinalFeatures,
    MetricAggregation,
    MetricDeviation,
    MetricTrend,
    TimeWindow,
    TrendDirection,
)
from ergosense.domain.monitoring import AnalysisResult, AnalysisState, FrameObservation
from ergosense.domain.session import (
    MonitoringSession,
    SessionObservation,
    SessionStatus,
)


def observed_at(seconds: int) -> datetime:
    return datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def make_session(start_seconds=0, ended_seconds=None):
    return MonitoringSession(
        session_id=uuid4(),
        started_at=observed_at(start_seconds),
        status=SessionStatus.ENDED
        if ended_seconds is not None
        else SessionStatus.ACTIVE,
        ended_at=None if ended_seconds is None else observed_at(ended_seconds),
    )


def make_baseline(confidence=BaselineConfidence.READY):
    return PersonalBaseline(
        confidence=confidence,
        eye=EyeBaseline(
            ear=MetricBaseline(0.28, 0.02, 12),
            perclos=MetricBaseline(0.10, 0.02, 12),
        ),
        head=HeadBaseline(
            pitch=MetricBaseline(3.0, 1.0, 12),
            yaw=MetricBaseline(2.0, 1.0, 12),
            roll=MetricBaseline(1.5, 1.0, 12),
        ),
    )


def make_observation(seconds: int, session_id, **metrics):
    observation = FrameObservation(
        timestamp=float(seconds),
        observed_at=observed_at(seconds),
        face_detected=metrics.get("face_detected", True),
        ear=metrics.get("ear", 0.28),
        perclos=metrics.get("perclos", 0.10),
        perclos_ready=metrics.get("perclos_ready", True),
        pitch=metrics.get("pitch", 3.0),
        yaw=metrics.get("yaw", 2.0),
        roll=metrics.get("roll", 1.5),
    )
    return SessionObservation(
        session_id=session_id,
        captured_at=observed_at(seconds),
        elapsed_since_start=timedelta(seconds=seconds),
        result=AnalysisResult(observation=observation, state=AnalysisState()),
    )


def make_features(
    window: TimeWindow, score_hint=10.0, observation_count=6, session_id=None
):
    deviation = MetricDeviation(
        current=score_hint / 100.0,
        baseline_mean=0.1,
        absolute_difference=0.0,
        z_score=0.0,
        baseline_sample_count=12,
        baseline_available=True,
    )
    aggregation = MetricAggregation(
        mean=score_hint / 100.0,
        minimum=score_hint / 100.0,
        maximum=score_hint / 100.0,
        standard_deviation=0.0,
        sample_count=observation_count,
    )
    trend = MetricTrend(
        direction=TrendDirection.STABLE, slope=0.0, sample_count=observation_count
    )
    end = observed_at(1000)
    return LongitudinalFeatures(
        session_id=session_id,
        window=window,
        window_started_at=end - window.duration,
        window_ended_at=end,
        elapsed_since_start=timedelta(minutes=20),
        time_since_previous_observation=timedelta(seconds=5),
        continuous_time_since_session_start=timedelta(minutes=20),
        observation_count=observation_count,
        ear_aggregation=aggregation,
        perclos_aggregation=aggregation,
        pitch_aggregation=aggregation,
        yaw_aggregation=aggregation,
        roll_aggregation=aggregation,
        ear_deviation=deviation,
        perclos_deviation=deviation,
        pitch_deviation=deviation,
        yaw_deviation=deviation,
        roll_deviation=deviation,
        ear_trend=trend,
        perclos_trend=trend,
        pitch_trend=trend,
        yaw_trend=trend,
        roll_trend=trend,
        events=(),
    )


def make_state(score, confidence=80.0, trend=FatigueTrend.STABLE):
    explanation = f"score {score}"
    component = FatigueEvidence(
        source="test",
        score=score,
        strength=score,
        persistence=score,
        explanation=explanation,
    )
    return FatigueState(
        session_id=uuid4(),
        calculated_at=observed_at(1000),
        score=score,
        level=FatigueLevel.NORMAL
        if score < 20
        else FatigueLevel.MILD
        if score < 40
        else FatigueLevel.MODERATE
        if score < 60
        else FatigueLevel.HIGH
        if score < 80
        else FatigueLevel.VERY_HIGH,
        trend=trend,
        confidence=confidence,
        ocular_component=component,
        postural_component=component,
        temporal_component=component,
        convergence_component=component,
        reasons=(explanation,),
    )


class FakeFeatureService:
    def __init__(self, mapping):
        self.mapping = mapping

    def extract_features(self, observations, baseline, window, at=None):
        return self.mapping[window.name]


class FakeFatigueEngine:
    def __init__(self, mapping):
        self.mapping = mapping

    def evaluate(self, features, previous_state=None):
        return self.mapping[features.window.name]


def get_window_state(assessment, name):
    for item in assessment.window_states:
        if item.window.name == name:
            return item
    raise AssertionError(name)


def test_assessment_with_no_observations_marks_all_windows_insufficient():
    service = FatigueAssessmentService()
    session = make_session()

    assessment = service.assess(session, None, [], at=observed_at(40))

    assert assessment.current_state is None
    assert assessment.primary_window is None
    assert assessment.session_elapsed == timedelta(seconds=40)
    assert assessment.available_observations == 0
    assert assessment.assessment_confidence < 20
    assert all(
        item.status is WindowAssessmentStatus.INSUFFICIENT_DATA
        for item in assessment.window_states
    )


def test_primary_window_prefers_longest_available_window():
    service = FatigueAssessmentService()
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 901, 5)
    ]

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(900)
    )

    assert assessment.primary_window is not None
    assert assessment.primary_window.name == "15m"
    assert (
        get_window_state(assessment, "15m").status is WindowAssessmentStatus.AVAILABLE
    )


def test_short_session_does_not_fake_long_windows():
    service = FatigueAssessmentService()
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 41, 5)
    ]

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(40)
    )

    assert (
        get_window_state(assessment, "1m").status
        is WindowAssessmentStatus.INSUFFICIENT_DATA
    )
    assert (
        get_window_state(assessment, "5m").status
        is WindowAssessmentStatus.INSUFFICIENT_DATA
    )
    assert (
        get_window_state(assessment, "15m").status
        is WindowAssessmentStatus.INSUFFICIENT_DATA
    )


def test_assessment_uses_last_observation_when_at_is_missing():
    service = FatigueAssessmentService()
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in (0, 60, 120)
    ]

    assessment = service.assess(session, make_baseline(), observations)

    assert assessment.calculated_at == observed_at(120)
    assert assessment.session_elapsed == timedelta(seconds=120)


def test_assessment_handles_missing_and_calibrating_baseline_conservatively():
    service = FatigueAssessmentService()
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 301, 5)
    ]

    no_baseline = service.assess(session, None, observations, at=observed_at(300))
    calibrating = service.assess(
        session,
        make_baseline(confidence=BaselineConfidence.CALIBRATING),
        observations,
        at=observed_at(300),
    )
    ready = service.assess(session, make_baseline(), observations, at=observed_at(300))

    assert not no_baseline.baseline_available
    assert not calibrating.baseline_available
    assert ready.baseline_available
    assert no_baseline.assessment_confidence < ready.assessment_confidence
    assert calibrating.assessment_confidence < ready.assessment_confidence


def test_consolidation_keeps_longer_context_as_primary_signal():
    windows = (
        TimeWindow.one_minute(),
        TimeWindow.five_minutes(),
        TimeWindow.fifteen_minutes(),
    )
    config = FatigueAssessmentConfig(windows=windows)
    features = {
        "1m": make_features(windows[0], score_hint=10.0),
        "5m": make_features(windows[1], score_hint=30.0),
        "15m": make_features(windows[2], score_hint=50.0),
    }
    states = {"1m": make_state(10), "5m": make_state(30), "15m": make_state(50)}
    service = FatigueAssessmentService(
        feature_service=FakeFeatureService(features),
        fatigue_engine=FakeFatigueEngine(states),
        config=config,
    )
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 901, 5)
    ]

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(900)
    )

    assert assessment.primary_window.name == "15m"
    assert assessment.current_state is not None
    assert assessment.current_state.level is FatigueLevel.MODERATE
    assert 40 <= assessment.current_state.score <= 50


def test_recent_spike_does_not_immediately_dominate_when_longer_windows_are_normal():
    windows = (
        TimeWindow.one_minute(),
        TimeWindow.five_minutes(),
        TimeWindow.fifteen_minutes(),
    )
    features = {
        name: make_features(window)
        for name, window in (
            ("1m", windows[0]),
            ("5m", windows[1]),
            ("15m", windows[2]),
        )
    }
    states = {"1m": make_state(75), "5m": make_state(10), "15m": make_state(10)}
    service = FatigueAssessmentService(
        feature_service=FakeFeatureService(features),
        fatigue_engine=FakeFatigueEngine(states),
        config=FatigueAssessmentConfig(windows=windows),
    )
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 901, 5)
    ]

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(900)
    )

    assert assessment.current_state is not None
    assert assessment.current_state.score < 30
    assert assessment.current_state.level in {FatigueLevel.NORMAL, FatigueLevel.MILD}


def test_consistent_window_states_stay_consistent_after_consolidation():
    windows = (
        TimeWindow.one_minute(),
        TimeWindow.five_minutes(),
        TimeWindow.fifteen_minutes(),
    )
    features = {
        name: make_features(window, score_hint=50.0)
        for name, window in (
            ("1m", windows[0]),
            ("5m", windows[1]),
            ("15m", windows[2]),
        )
    }
    states = {"1m": make_state(50), "5m": make_state(50), "15m": make_state(50)}
    service = FatigueAssessmentService(
        feature_service=FakeFeatureService(features),
        fatigue_engine=FakeFatigueEngine(states),
        config=FatigueAssessmentConfig(windows=windows),
    )
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 901, 5)
    ]

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(900)
    )

    assert assessment.current_state is not None
    assert assessment.current_state.score == pytest.approx(50.0)
    assert assessment.current_state.level is FatigueLevel.MODERATE


def test_sustained_higher_long_windows_are_reflected_even_if_recent_window_is_normal():
    windows = (
        TimeWindow.one_minute(),
        TimeWindow.five_minutes(),
        TimeWindow.fifteen_minutes(),
    )
    features = {
        name: make_features(window, score_hint=70.0)
        for name, window in (
            ("1m", windows[0]),
            ("5m", windows[1]),
            ("15m", windows[2]),
        )
    }
    states = {"1m": make_state(10), "5m": make_state(70), "15m": make_state(70)}
    service = FatigueAssessmentService(
        feature_service=FakeFeatureService(features),
        fatigue_engine=FakeFatigueEngine(states),
        config=FatigueAssessmentConfig(windows=windows),
    )
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 901, 5)
    ]

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(900)
    )

    assert assessment.current_state is not None
    assert assessment.current_state.score >= 60
    assert assessment.current_state.level is FatigueLevel.HIGH


def test_assessment_trend_uses_previous_assessment_when_available():
    service = FatigueAssessmentService()
    session = make_session()
    observations = [
        make_observation(second, session.session_id, perclos=0.30, ear=0.20)
        for second in range(0, 901, 5)
    ]
    previous = service.assess(
        session, make_baseline(), observations, at=observed_at(600)
    )
    current = service.assess(
        session,
        make_baseline(),
        observations,
        at=observed_at(900),
        previous_assessment=previous,
    )

    assert current.trend in {
        AssessmentTrend.STABLE,
        AssessmentTrend.INCREASING,
        AssessmentTrend.DECREASING,
    }


def test_assessment_trend_is_unknown_without_previous_and_short_primary_window():
    service = FatigueAssessmentService()
    session = make_session()
    observations = [
        make_observation(second, session.session_id) for second in range(0, 61, 5)
    ]

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(60)
    )

    assert assessment.primary_window is not None
    assert assessment.primary_window.name == "1m"
    assert assessment.trend is AssessmentTrend.UNKNOWN


def test_assess_current_session_uses_injected_services():
    from ergosense.application.baseline_service import BaselineService
    from ergosense.application.session_service import SessionService

    session_service = SessionService(now=lambda: observed_at(0))
    baseline_service = BaselineService(now=lambda: observed_at(0))
    session = session_service.start_session(started_at=observed_at(0))
    baseline_service.start_calibration(session=session)
    for second in range(0, 301, 5):
        observation = FrameObservation(
            timestamp=float(second),
            observed_at=observed_at(second),
            face_detected=True,
            ear=0.28,
            perclos=0.10,
            perclos_ready=True,
            pitch=3.0,
            yaw=2.0,
            roll=1.5,
        )
        result = AnalysisResult(
            observation=observation,
            state=AnalysisState(),
            session_id=session.session_id,
        )
        session_service.register_observation(result)
        baseline_service.add_observation(observation)

    service = FatigueAssessmentService(
        session_service=session_service,
        baseline_service=baseline_service,
    )

    assessment = service.assess_current_session(at=observed_at(300))

    assert assessment.session_id == session.session_id
    assert assessment.available_observations > 0


def test_full_flow_integration_without_camera_or_opencv():
    session = make_session()
    observations = [
        make_observation(second, session.session_id, ear=0.24, perclos=0.18, pitch=8.0)
        for second in range(0, 901, 5)
    ]
    service = FatigueAssessmentService(
        feature_service=LongitudinalFeatureService(),
        fatigue_engine=FatigueEngine(now=lambda: observed_at(900)),
    )

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(900)
    )

    assert assessment.current_state is not None
    assert assessment.primary_window is not None
    assert assessment.current_state.level in {
        FatigueLevel.MILD,
        FatigueLevel.MODERATE,
        FatigueLevel.HIGH,
        FatigueLevel.VERY_HIGH,
        FatigueLevel.NORMAL,
    }
    assert 0 <= assessment.current_state.score <= 100
    assert 0 <= assessment.assessment_confidence <= 100


def test_invalid_metric_values_do_not_break_assessment_flow():
    session = make_session()
    observations = [
        make_observation(0, session.session_id, ear=None, perclos=None),
        make_observation(5, session.session_id, ear=float("nan"), pitch=float("inf")),
        make_observation(10, session.session_id, ear=0.25, perclos=0.12, pitch=4.0),
    ]
    service = FatigueAssessmentService(
        feature_service=LongitudinalFeatureService(),
        fatigue_engine=FatigueEngine(now=lambda: observed_at(10)),
    )

    assessment = service.assess(
        session, make_baseline(), observations, at=observed_at(60)
    )

    assert assessment.available_observations == 3
    assert get_window_state(assessment, "1m").status is WindowAssessmentStatus.AVAILABLE
