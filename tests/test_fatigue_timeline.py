from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from ergosense.application.fatigue_timeline_service import FatigueTimelineService
from ergosense.domain.assessment import (
    AssessmentTrend,
    FatigueAssessment,
    WindowAssessmentStatus,
    WindowFatigueAssessment,
)
from ergosense.domain.fatigue import (
    FatigueEvidence,
    FatigueLevel,
    FatigueState,
    FatigueTrend,
)
from ergosense.domain.fatigue_timeline import FatigueTimelineConfig
from ergosense.domain.longitudinal import TimeWindow


def observed_at(minutes: int) -> datetime:
    return datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=minutes
    )


def make_state(
    score: float, *, trend: FatigueTrend = FatigueTrend.STABLE
) -> FatigueState:
    component = FatigueEvidence(
        source="test",
        score=score,
        strength=score,
        persistence=score,
        explanation=f"score {score}",
    )
    if score < 20:
        level = FatigueLevel.NORMAL
    elif score < 40:
        level = FatigueLevel.MILD
    elif score < 60:
        level = FatigueLevel.MODERATE
    elif score < 80:
        level = FatigueLevel.HIGH
    else:
        level = FatigueLevel.VERY_HIGH
    return FatigueState(
        session_id=uuid4(),
        calculated_at=observed_at(0),
        score=score,
        level=level,
        trend=trend,
        confidence=80.0,
        ocular_component=component,
        postural_component=component,
        temporal_component=component,
        convergence_component=component,
        reasons=(f"score {score}",),
    )


def make_assessment(session_id, minute: int, score: float | None) -> FatigueAssessment:
    state = None if score is None else make_state(score)
    window = TimeWindow.one_minute()
    return FatigueAssessment(
        session_id=session_id,
        calculated_at=observed_at(minute),
        current_state=state,
        window_states=(
            WindowFatigueAssessment(
                window=window,
                status=WindowAssessmentStatus.AVAILABLE
                if state is not None
                else WindowAssessmentStatus.INSUFFICIENT_DATA,
                state=state,
                observation_count=3 if state is not None else 0,
            ),
        ),
        primary_window=window if state is not None else None,
        available_observations=3 if state is not None else 0,
        baseline_available=True,
        session_elapsed=timedelta(minutes=minute),
        assessment_confidence=80.0 if state is not None else 10.0,
        trend=AssessmentTrend.UNKNOWN,
    )


def test_empty_timeline_properties_are_conservative():
    timeline = FatigueTimelineService().create_timeline("session-1")

    assert timeline.is_empty
    assert timeline.first is None
    assert timeline.latest is None
    assert timeline.duration == timedelta(0)
    assert timeline.peak_score is None
    assert timeline.time_to_first_fatigue() is None
    assert timeline.first_sustained_fatigue() is None
    assert timeline.transitions() == ()
    assert timeline.trend() is AssessmentTrend.UNKNOWN


def test_add_first_assessment_creates_first_point():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    assessment = make_assessment(session_id, 0, 10)

    timeline = timeline.add(assessment)

    assert not timeline.is_empty
    assert timeline.first is not None
    assert timeline.latest is not None
    assert timeline.first.calculated_at == observed_at(0)
    assert timeline.latest.score == 10


def test_add_assessments_keeps_chronological_order():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)

    timeline = timeline.add(make_assessment(session_id, 0, 10))
    timeline = timeline.add(make_assessment(session_id, 5, 25))

    assert tuple(point.calculated_at for point in timeline.points) == (
        observed_at(0),
        observed_at(5),
    )


def test_rejects_assessment_from_another_session():
    timeline = (
        FatigueTimelineService().create_timeline("A").add(make_assessment("A", 0, 10))
    )

    with pytest.raises(ValueError, match="session_id"):
        timeline.add(make_assessment("B", 5, 20))


def test_rejects_timestamp_earlier_than_latest():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    timeline = timeline.add(make_assessment(session_id, 10, 10))

    with pytest.raises(ValueError, match="earlier"):
        timeline.add(make_assessment(session_id, 5, 20))


def test_rejects_duplicate_timestamp():
    session_id = uuid4()
    assessment = make_assessment(session_id, 10, 10)
    timeline = FatigueTimelineService().create_timeline(session_id).add(assessment)

    with pytest.raises(ValueError, match="duplicate"):
        timeline.add(make_assessment(session_id, 10, 20))


def test_duration_latest_first_and_peak_score_are_correct():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    for minute, score in ((0, 10), (5, 25), (10, 55)):
        timeline = timeline.add(make_assessment(session_id, minute, score))

    assert timeline.first.calculated_at == observed_at(0)
    assert timeline.latest.calculated_at == observed_at(10)
    assert timeline.duration == timedelta(minutes=10)
    assert timeline.peak_score == 55


def test_time_to_first_fatigue_uses_first_non_normal_point():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    for minute, score in ((0, 10), (5, 15), (10, 25), (15, 30)):
        timeline = timeline.add(make_assessment(session_id, minute, score))

    assert timeline.time_to_first_fatigue() == timedelta(minutes=10)


def test_first_sustained_fatigue_requires_consecutive_points():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    for minute, score in ((0, 10), (5, 25), (10, 15), (15, 25), (20, 30)):
        timeline = timeline.add(make_assessment(session_id, minute, score))

    sustained = timeline.first_sustained_fatigue()

    assert sustained is not None
    assert sustained.calculated_at == observed_at(20)


def test_first_sustained_fatigue_respects_configured_point_count():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    for minute, score in ((0, 10), (5, 25), (10, 30), (15, 35)):
        timeline = timeline.add(make_assessment(session_id, minute, score))

    sustained = timeline.first_sustained_fatigue(
        FatigueTimelineConfig(sustained_fatigue_points=3)
    )

    assert sustained is not None
    assert sustained.calculated_at == observed_at(15)


def test_detects_level_transition_normal_to_mild():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    timeline = timeline.add(make_assessment(session_id, 0, 10))
    timeline = timeline.add(make_assessment(session_id, 5, 25))

    transitions = timeline.transitions()

    assert len(transitions) == 1
    assert transitions[0].from_level is FatigueLevel.NORMAL
    assert transitions[0].to_level is FatigueLevel.MILD


def test_detects_multiple_transitions_and_ignores_same_level_changes():
    session_id = uuid4()
    timeline = FatigueTimelineService().create_timeline(session_id)
    for minute, score in (
        (0, 10),
        (5, 15),
        (10, 25),
        (15, 45),
        (20, 48),
        (25, 70),
        (30, 35),
    ):
        timeline = timeline.add(make_assessment(session_id, minute, score))

    transitions = timeline.transitions()

    assert [(item.from_level, item.to_level) for item in transitions] == [
        (FatigueLevel.NORMAL, FatigueLevel.MILD),
        (FatigueLevel.MILD, FatigueLevel.MODERATE),
        (FatigueLevel.MODERATE, FatigueLevel.HIGH),
        (FatigueLevel.HIGH, FatigueLevel.MILD),
    ]


def test_trend_is_unknown_with_less_than_two_scored_points():
    session_id = uuid4()
    timeline = (
        FatigueTimelineService()
        .create_timeline(session_id)
        .add(make_assessment(session_id, 0, None))
    )

    assert timeline.trend() is AssessmentTrend.UNKNOWN


def test_trend_can_be_increasing_decreasing_or_stable():
    session_id = uuid4()
    service = FatigueTimelineService()

    increasing = service.build_timeline(
        [make_assessment(session_id, 0, 10), make_assessment(session_id, 5, 20)]
    )
    decreasing = service.build_timeline(
        [make_assessment(session_id, 0, 40), make_assessment(session_id, 5, 20)]
    )
    stable = service.build_timeline(
        [make_assessment(session_id, 0, 40), make_assessment(session_id, 5, 43)]
    )

    assert increasing.trend() is AssessmentTrend.INCREASING
    assert decreasing.trend() is AssessmentTrend.DECREASING
    assert stable.trend() is AssessmentTrend.STABLE


def test_points_cannot_be_mutated_accidentally_from_outside():
    session_id = uuid4()
    timeline = FatigueTimelineService().build_timeline(
        [make_assessment(session_id, 0, 10), make_assessment(session_id, 5, 20)]
    )

    with pytest.raises(AttributeError):
        timeline.points.append("nope")


def test_build_timeline_rejects_invalid_sequences():
    service = FatigueTimelineService()
    session_id = uuid4()

    with pytest.raises(ValueError, match="earlier"):
        service.build_timeline(
            [make_assessment(session_id, 5, 10), make_assessment(session_id, 0, 20)]
        )


def test_build_timeline_and_service_helpers_work_for_ordered_input():
    service = FatigueTimelineService()
    session_id = uuid4()
    timeline = service.build_timeline(
        [make_assessment(session_id, 0, 10), make_assessment(session_id, 5, 20)]
    )

    assert service.get_latest(timeline).calculated_at == observed_at(5)
    assert service.get_duration(timeline) == timedelta(minutes=5)
