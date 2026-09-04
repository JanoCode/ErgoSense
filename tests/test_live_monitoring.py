from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from ergosense.application.live_monitoring import (
    LiveFrameAnalyzer,
    LiveMonitoringService,
    build_alerts,
)
from ergosense.domain.monitoring import AnalysisResult, AnalysisState
from ergosense.domain.monitoring import FrameObservation
from ergosense.domain.session import MonitoringSession, SessionStatus
from ergosense.infrastructure.live_monitoring import LiveFrameSample


class FakeScorer:
    def __init__(self):
        self.perclos_ready = False
        self.missing_calls = []
        self.perclos_calls = []
        self.eval_calls = []

    def mark_face_missing(self, timestamp):
        self.missing_calls.append(timestamp)
        self.perclos_ready = True

    def get_rolling_PERCLOS(self, timestamp, ear):
        self.perclos_calls.append((timestamp, ear))
        self.perclos_ready = True
        return True, 0.42

    def eval_scores(self, timestamp, ear, gaze, roll, pitch, yaw):
        self.eval_calls.append((timestamp, ear, gaze, roll, pitch, yaw))
        return False, True, True


def make_observation(timestamp=1.0, observed_seconds=1.0, **kwargs):
    observed_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=observed_seconds
    )
    data = {
        "timestamp": timestamp,
        "observed_at": observed_at,
        "face_detected": False,
    }
    data.update(kwargs)
    return FrameObservation(**data)


def test_build_alerts_preserves_existing_alert_order():
    assert build_alerts(
        tired=True,
        asleep=True,
        looking_away=True,
        distracted=True,
    ) == ("TIRED", "ASLEEP", "LOOKING AWAY", "DISTRACTED")


def test_analyzer_marks_missing_face_without_generating_alerts():
    scorer = FakeScorer()
    analyzer = LiveFrameAnalyzer(scorer)

    result = analyzer.analyze(make_observation(timestamp=5.0, observed_seconds=5.0))

    assert scorer.missing_calls == [5.0]
    assert scorer.perclos_calls == []
    assert result.alerts == ()
    assert not result.state.tired
    assert result.perclos is None
    assert result.perclos_ready


def test_analyzer_uses_existing_scorer_outputs_for_results():
    scorer = FakeScorer()
    analyzer = LiveFrameAnalyzer(scorer)
    observation = FrameObservation(
        timestamp=7.0,
        observed_at=datetime(2026, 1, 1, 0, 0, 7, tzinfo=timezone.utc),
        face_detected=True,
        ear=0.1,
        gaze=0.3,
        roll=25.0,
        pitch=0.0,
        yaw=0.0,
    )

    result = analyzer.analyze(observation)

    assert scorer.perclos_calls == [(7.0, 0.1)]
    assert scorer.eval_calls == [(7.0, 0.1, 0.3, 25.0, 0.0, 0.0)]
    assert result.state.tired
    assert not result.state.asleep
    assert result.state.looking_away
    assert result.state.distracted
    assert result.alerts == ("TIRED", "LOOKING AWAY", "DISTRACTED")
    assert result.perclos == 0.42
    assert result.perclos_ready


class FakeSource:
    def stream(self):
        yield LiveFrameSample(
            frame="frame-1",
            observation=make_observation(timestamp=1.0, observed_seconds=1.0, fps=30.0),
            processing_started_at=10.0,
        )


class FakeSessionService:
    def __init__(self, session):
        self.session = session
        self.recorded = []

    def get_session(self):
        return self.session

    def get_status(self):
        return self.session.status

    def register_observation(self, result):
        self.recorded.append(result)


def test_service_adds_processing_time_to_emitted_observation(monkeypatch):
    source = FakeSource()

    class Analyzer:
        def analyze(self, observation):
            return AnalysisResult(observation=observation, state=AnalysisState())

    service = LiveMonitoringService(source, Analyzer())
    monkeypatch.setattr(
        "ergosense.application.live_monitoring.time.perf_counter", lambda: 10.012
    )

    frame, result = next(service.stream())

    assert frame == "frame-1"
    assert result.observation.fps == 30.0
    assert result.observation.processing_ms == pytest.approx(12.0)


def test_service_attaches_session_id_and_registers_only_sampled_results(monkeypatch):
    source = FakeSource()
    session = MonitoringSession(
        session_id=uuid4(),
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=SessionStatus.ACTIVE,
    )
    session_service = FakeSessionService(session)

    class Analyzer:
        def analyze(self, observation):
            return AnalysisResult(observation=observation, state=AnalysisState())

    class Sampler:
        def __init__(self):
            self.calls = []

        def should_sample(self, observed_at):
            self.calls.append(observed_at)
            return True

    sampler = Sampler()
    service = LiveMonitoringService(
        source,
        Analyzer(),
        session_service=session_service,
        observation_sampler=sampler,
    )
    monkeypatch.setattr(
        "ergosense.application.live_monitoring.time.perf_counter", lambda: 10.012
    )

    _, result = next(service.stream())

    assert result.session_id == session.session_id
    assert sampler.calls == [result.observation.observed_at]
    assert session_service.recorded == [result]
