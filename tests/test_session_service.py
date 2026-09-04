from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from ergosense.application.session_service import ObservationSampler, SessionService
from ergosense.domain.monitoring import AnalysisResult, AnalysisState, FrameObservation
from ergosense.domain.session import SessionStatus


def observed_at(seconds: int) -> datetime:
    return datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def make_result(seconds: int) -> AnalysisResult:
    observation = FrameObservation(
        timestamp=float(seconds),
        observed_at=observed_at(seconds),
        face_detected=True,
        ear=0.2,
        gaze=0.1,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )
    return AnalysisResult(observation=observation, state=AnalysisState())


def test_start_session_creates_active_session():
    service = SessionService(now=lambda: observed_at(0))

    session = service.start_session()

    assert isinstance(session.session_id, UUID)
    assert session.started_at == observed_at(0)
    assert session.status is SessionStatus.ACTIVE
    assert session.ended_at is None


def test_end_session_marks_session_as_ended():
    service = SessionService(now=lambda: observed_at(0))
    service.start_session()

    ended = service.end_session(ended_at=observed_at(10))

    assert ended.status is SessionStatus.ENDED
    assert ended.ended_at == observed_at(10)


def test_cannot_start_new_session_while_one_is_active():
    service = SessionService(now=lambda: observed_at(0))
    service.start_session()

    with pytest.raises(ValueError, match="active"):
        service.start_session()


def test_session_rejects_end_before_start():
    with pytest.raises(ValueError, match="earlier"):
        service = SessionService(now=lambda: observed_at(0))
        service.start_session(started_at=observed_at(10))
        service.end_session(ended_at=observed_at(5))


def test_duration_is_calculated_for_active_session_without_real_clock():
    service = SessionService(now=lambda: observed_at(12))
    service.start_session(started_at=observed_at(0))

    assert service.get_duration() == timedelta(seconds=12)


def test_duration_is_calculated_for_ended_session():
    service = SessionService(now=lambda: observed_at(0))
    service.start_session(started_at=observed_at(0))
    service.end_session(ended_at=observed_at(30))

    assert service.get_duration() == timedelta(seconds=30)


def test_register_observation_associates_result_with_active_session():
    service = SessionService(now=lambda: observed_at(0))
    session = service.start_session(started_at=observed_at(0))

    recorded = service.register_observation(make_result(5))

    assert recorded.session_id == session.session_id
    assert recorded.captured_at == observed_at(5)
    assert recorded.elapsed_since_start == timedelta(seconds=5)
    assert recorded.result.session_id == session.session_id
    assert service.list_observations() == (recorded,)


def test_register_observation_requires_active_session():
    service = SessionService(now=lambda: observed_at(0))

    with pytest.raises(ValueError, match="active session"):
        service.register_observation(make_result(5))


def test_register_observation_rejects_timestamp_before_session_start():
    service = SessionService(now=lambda: observed_at(0))
    service.start_session(started_at=observed_at(10))

    with pytest.raises(ValueError, match="earlier"):
        service.register_observation(make_result(5))


def test_sampler_does_not_emit_before_interval():
    sampler = ObservationSampler(interval=timedelta(seconds=5))
    sampler.start(observed_at(0))

    assert not sampler.should_sample(observed_at(4))


def test_sampler_emits_once_interval_has_elapsed():
    sampler = ObservationSampler(interval=timedelta(seconds=5))
    sampler.start(observed_at(0))

    assert sampler.should_sample(observed_at(5))


def test_sampler_respects_repeated_intervals_without_real_clock():
    sampler = ObservationSampler(interval=timedelta(seconds=5))
    sampler.start(observed_at(0))

    assert sampler.should_sample(observed_at(12))
    assert not sampler.should_sample(observed_at(14))
    assert sampler.should_sample(observed_at(15))
