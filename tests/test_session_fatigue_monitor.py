from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from ergosense.application.baseline_service import BaselineService
from ergosense.application.fatigue_assessment_service import FatigueAssessmentService
from ergosense.application.fatigue_timeline_service import FatigueTimelineService
from ergosense.application.session_fatigue_monitor import (
    SessionFatigueMonitor,
    SessionFatigueMonitorConfig,
)
from ergosense.application.session_service import SessionService
from ergosense.domain.monitoring import AnalysisResult, AnalysisState, FrameObservation
from ergosense.domain.session import SessionObservation


def observed_at(seconds: int) -> datetime:
    return datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def make_observation(session_id, seconds: int) -> SessionObservation:
    frame_observation = FrameObservation(
        timestamp=float(seconds),
        observed_at=observed_at(seconds),
        face_detected=True,
        ear=0.28,
        perclos=0.10,
        perclos_ready=True,
        pitch=3.0,
        yaw=2.0,
        roll=1.5,
    )
    return SessionObservation(
        session_id=session_id,
        captured_at=observed_at(seconds),
        elapsed_since_start=timedelta(seconds=seconds),
        result=AnalysisResult(observation=frame_observation, state=AnalysisState()),
    )


def make_monitor(*, interval_seconds=30.0):
    session_service = SessionService(now=lambda: observed_at(0))
    baseline_service = BaselineService(now=lambda: observed_at(0))
    assessment_service = FatigueAssessmentService(
        session_service=session_service,
        baseline_service=baseline_service,
    )
    timeline_service = FatigueTimelineService()
    monitor = SessionFatigueMonitor(
        session_service=session_service,
        assessment_service=assessment_service,
        timeline_service=timeline_service,
        config=SessionFatigueMonitorConfig(
            assessment_interval_seconds=interval_seconds
        ),
    )
    return monitor, session_service, baseline_service


def test_monitor_starts_session_and_initializes_timeline():
    monitor, _, _ = make_monitor()

    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())

    assert monitor.is_running()
    assert monitor.timeline() is not None
    assert monitor.timeline().session_id == session.session_id
    assert monitor.current_assessment() is None


def test_process_observation_accepts_matching_session_and_reuses_services():
    monitor, session_service, baseline_service = make_monitor()
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())

    result = monitor.process_observation(make_observation(session.session_id, 5))

    assert result is None
    assert len(session_service.list_observations()) == 1
    assert baseline_service.get_baseline(at=observed_at(5)).eye.ear.sample_count == 1


def test_rejects_observation_from_other_session():
    monitor, _, _ = make_monitor()
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())

    with pytest.raises(ValueError, match="session_id"):
        monitor.process_observation(make_observation(uuid4(), 5))

    assert monitor.timeline().session_id == session.session_id


def test_does_not_generate_assessment_before_interval():
    monitor, _, _ = make_monitor(interval_seconds=30.0)
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())

    for second in (5, 10, 15, 20, 25):
        assert (
            monitor.process_observation(make_observation(session.session_id, second))
            is None
        )

    assert monitor.timeline().is_empty


def test_generates_first_assessment_at_interval_and_keeps_insufficient_data_explicit():
    monitor, _, _ = make_monitor(interval_seconds=30.0)
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())
    for second in (5, 10, 15, 20, 25):
        monitor.process_observation(make_observation(session.session_id, second))

    assessment = monitor.process_observation(make_observation(session.session_id, 30))

    assert assessment is not None
    assert assessment.current_state is None
    assert monitor.latest_assessment() is assessment
    assert len(monitor.timeline().points) == 1


def test_second_assessment_is_added_to_same_timeline_in_chronological_order():
    monitor, _, _ = make_monitor(interval_seconds=30.0)
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())

    for second in range(5, 65, 5):
        monitor.process_observation(make_observation(session.session_id, second))

    timeline = monitor.timeline()

    assert len(timeline.points) == 2
    assert timeline.points[0].calculated_at == observed_at(30)
    assert timeline.points[1].calculated_at == observed_at(60)
    assert timeline.session_id == session.session_id
    assert monitor.current_assessment() == timeline.latest.assessment


def test_latest_assessment_returns_latest_generated_assessment():
    monitor, _, _ = make_monitor(interval_seconds=30.0)
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())
    latest = None
    for second in range(5, 95, 5):
        assessment = monitor.process_observation(
            make_observation(session.session_id, second)
        )
        if assessment is not None:
            latest = assessment

    assert latest is not None
    assert monitor.latest_assessment() == latest
    assert monitor.timeline().latest.assessment == latest


def test_stop_prevents_new_observations_and_keeps_existing_timeline():
    monitor, _, _ = make_monitor()
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())
    for second in range(5, 35, 5):
        monitor.process_observation(make_observation(session.session_id, second))

    timeline = monitor.stop()

    assert not monitor.is_running()
    assert timeline is monitor.timeline()
    with pytest.raises(ValueError, match="not running"):
        monitor.process_observation(make_observation(session.session_id, 35))


def test_assessment_interval_is_configurable():
    monitor, _, _ = make_monitor(interval_seconds=15.0)
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())

    assert monitor.process_observation(make_observation(session.session_id, 5)) is None
    assert monitor.process_observation(make_observation(session.session_id, 10)) is None
    assessment = monitor.process_observation(make_observation(session.session_id, 15))

    assert assessment is not None
    assert assessment.calculated_at == observed_at(15)


def test_monitor_uses_observation_timestamps_deterministically():
    monitor, _, _ = make_monitor(interval_seconds=30.0)
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())
    for second in (7, 14, 21, 28):
        monitor.process_observation(make_observation(session.session_id, second))

    assert (
        monitor.process_observation(make_observation(session.session_id, 35))
        is not None
    )
    assert monitor.timeline().latest.calculated_at == observed_at(35)


def test_monitor_does_not_reset_timeline_for_each_assessment():
    monitor, _, _ = make_monitor(interval_seconds=30.0)
    session = monitor.start_session(started_at=observed_at(0), session_id=uuid4())
    for second in range(5, 95, 5):
        monitor.process_observation(make_observation(session.session_id, second))

    timeline = monitor.timeline()

    assert len(timeline.points) == 3
    assert [point.calculated_at for point in timeline.points] == [
        observed_at(30),
        observed_at(60),
        observed_at(90),
    ]
    assert all(
        point.assessment.session_id == session.session_id for point in timeline.points
    )
