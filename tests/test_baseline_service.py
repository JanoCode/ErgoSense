from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from ergosense.application.baseline_service import BaselineService
from ergosense.domain.baseline import BaselineConfidence, PersonalBaseline
from ergosense.domain.monitoring import FrameObservation
from ergosense.domain.session import MonitoringSession, SessionStatus


def observed_at(seconds: int) -> datetime:
    return datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def make_observation(seconds: int, **kwargs) -> FrameObservation:
    data = {
        "timestamp": float(seconds),
        "observed_at": observed_at(seconds),
        "face_detected": True,
        "ear": 0.2,
        "perclos": 0.1,
        "perclos_ready": True,
        "gaze": 0.05,
        "pitch": 1.0,
        "yaw": 2.0,
        "roll": 3.0,
    }
    data.update(kwargs)
    return FrameObservation(**data)


def test_create_empty_baseline():
    service = BaselineService(now=lambda: observed_at(0))

    baseline = service.get_baseline()

    assert isinstance(baseline, PersonalBaseline)
    assert baseline.confidence is BaselineConfidence.INSUFFICIENT_DATA
    assert baseline.eye.ear.sample_count == 0
    assert baseline.head.pitch.sample_count == 0


def test_adding_valid_observation_updates_counts_and_means():
    service = BaselineService(now=lambda: observed_at(0))
    service.start_calibration(started_at=observed_at(0))

    baseline = service.add_observation(make_observation(5, ear=0.24, pitch=4.0))

    assert baseline.eye.ear.sample_count == 1
    assert baseline.eye.ear.mean == pytest.approx(0.24)
    assert baseline.head.pitch.mean == pytest.approx(4.0)


def test_standard_deviation_is_calculated_incrementally_and_deterministically():
    service = BaselineService(now=lambda: observed_at(0))
    service.start_calibration(started_at=observed_at(0))

    service.add_observation(make_observation(5, ear=0.20))
    baseline = service.add_observation(make_observation(10, ear=0.24))

    assert baseline.eye.ear.mean == pytest.approx(0.22)
    assert baseline.eye.ear.standard_deviation == pytest.approx(0.0282842712474619)


def test_observations_without_face_are_ignored():
    service = BaselineService(now=lambda: observed_at(0))
    service.start_calibration(started_at=observed_at(0))

    baseline = service.add_observation(
        make_observation(5, face_detected=False, ear=0.24)
    )

    assert baseline.eye.ear.sample_count == 0
    assert baseline.confidence is BaselineConfidence.INSUFFICIENT_DATA


def test_invalid_values_are_ignored_per_metric():
    service = BaselineService(now=lambda: observed_at(0))
    service.start_calibration(started_at=observed_at(0))

    baseline = service.add_observation(
        make_observation(5, ear=float("nan"), perclos=float("inf"), pitch=None)
    )

    assert baseline.eye.ear.sample_count == 0
    assert baseline.eye.perclos.sample_count == 0
    assert baseline.head.pitch.sample_count == 0
    assert baseline.head.yaw.sample_count == 1
    assert baseline.head.roll.sample_count == 1


def test_calibration_is_incomplete_before_duration_and_sample_threshold():
    service = BaselineService(
        now=lambda: observed_at(0),
        calibration_duration=timedelta(seconds=120),
        min_samples_per_metric=2,
    )
    service.start_calibration(started_at=observed_at(0))
    service.add_observation(make_observation(5))

    assert not service.is_ready(at=observed_at(30))
    assert (
        service.get_baseline(at=observed_at(30)).confidence
        is BaselineConfidence.CALIBRATING
    )


def test_calibration_becomes_ready_when_duration_and_samples_are_sufficient():
    service = BaselineService(
        now=lambda: observed_at(0),
        calibration_duration=timedelta(seconds=120),
        min_samples_per_metric=2,
    )
    service.start_calibration(started_at=observed_at(0))
    service.add_observation(make_observation(5))
    service.add_observation(make_observation(125, ear=0.22, perclos=0.12))

    baseline = service.get_baseline(at=observed_at(125))

    assert service.is_ready(at=observed_at(125))
    assert baseline.confidence is BaselineConfidence.READY


def test_finish_calibration_sets_completion_time_and_blocks_further_updates():
    service = BaselineService(
        now=lambda: observed_at(0),
        calibration_duration=timedelta(seconds=10),
        min_samples_per_metric=1,
    )
    service.start_calibration(started_at=observed_at(0))
    service.add_observation(make_observation(10))

    baseline = service.finish_calibration(completed_at=observed_at(10))

    assert baseline.completed_at == observed_at(10)
    assert baseline.confidence is BaselineConfidence.READY
    with pytest.raises(ValueError, match="already been finished"):
        service.add_observation(make_observation(15))


def test_cannot_add_observation_before_starting_calibration():
    service = BaselineService(now=lambda: observed_at(0))

    with pytest.raises(ValueError, match="not been started"):
        service.add_observation(make_observation(5))


def test_perclos_is_ignored_until_ready():
    service = BaselineService(now=lambda: observed_at(0))
    service.start_calibration(started_at=observed_at(0))

    baseline = service.add_observation(
        make_observation(5, perclos=0.5, perclos_ready=False)
    )

    assert baseline.eye.perclos.sample_count == 0


def test_calibration_uses_session_context_when_provided():
    service = BaselineService(now=lambda: observed_at(0))
    session = MonitoringSession(
        session_id=uuid4(),
        started_at=observed_at(0),
        status=SessionStatus.ACTIVE,
    )

    baseline = service.start_calibration(session=session)

    assert baseline.session_id == session.session_id
    assert baseline.started_at == session.started_at


def test_deterministic_results_match_for_same_inputs():
    service_a = BaselineService(now=lambda: observed_at(0), min_samples_per_metric=1)
    service_b = BaselineService(now=lambda: observed_at(0), min_samples_per_metric=1)
    for service in (service_a, service_b):
        service.start_calibration(started_at=observed_at(0))
        service.add_observation(make_observation(5, ear=0.20, pitch=1.0))
        service.add_observation(make_observation(10, ear=0.30, pitch=3.0))

    baseline_a = service_a.get_baseline(at=observed_at(120))
    baseline_b = service_b.get_baseline(at=observed_at(120))

    assert baseline_a.eye.ear.mean == pytest.approx(baseline_b.eye.ear.mean)
    assert baseline_a.eye.ear.standard_deviation == pytest.approx(
        baseline_b.eye.ear.standard_deviation
    )
    assert baseline_a.head.pitch.mean == pytest.approx(baseline_b.head.pitch.mean)


def test_invalid_only_data_does_not_create_absurd_metric_counts():
    service = BaselineService(now=lambda: observed_at(0), min_samples_per_metric=1)
    service.start_calibration(started_at=observed_at(0))
    service.add_observation(
        make_observation(
            5,
            ear=float("nan"),
            perclos=float("nan"),
            pitch=float("inf"),
            yaw=None,
            roll=float("-inf"),
        )
    )

    baseline = service.get_baseline(at=observed_at(120))

    assert baseline.eye.ear.sample_count == 0
    assert baseline.eye.perclos.sample_count == 0
    assert baseline.head.pitch.sample_count == 0
    assert baseline.head.yaw.sample_count == 0
    assert baseline.head.roll.sample_count == 0
    assert baseline.confidence is BaselineConfidence.INSUFFICIENT_DATA
