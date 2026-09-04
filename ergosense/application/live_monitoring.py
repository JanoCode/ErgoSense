from dataclasses import replace
import time
from collections.abc import Iterator
from typing import Any, Protocol

from ergosense.domain.monitoring import AnalysisResult, AnalysisState, FrameObservation
from ergosense.domain.session import MonitoringSession, SessionStatus


class SupportsAttentionScoring(Protocol):
    perclos_ready: bool

    def mark_face_missing(self, timestamp: float, /) -> None: ...

    def get_rolling_PERCLOS(
        self, timestamp: float, ear: float | None, /
    ) -> tuple[bool, float]: ...

    def eval_scores(
        self,
        timestamp: float,
        ear: float | None,
        gaze: float | None,
        roll: float | None,
        pitch: float | None,
        yaw: float | None,
        /,
    ) -> tuple[bool, bool, bool]: ...


class SupportsObservationSource(Protocol):
    def stream(self) -> Iterator[Any]: ...


class SupportsSessionService(Protocol):
    def get_session(self) -> MonitoringSession | None: ...

    def get_status(self) -> SessionStatus | None: ...

    def register_observation(self, result: AnalysisResult) -> Any: ...


class SupportsObservationSampler(Protocol):
    def should_sample(self, observed_at) -> bool: ...


class SupportsBaselineService(Protocol):
    def add_observation(self, observation: FrameObservation) -> Any: ...


class SupportsResultAnalyzer(Protocol):
    def analyze(self, observation: FrameObservation) -> AnalysisResult: ...


def build_alerts(
    *, tired: bool, asleep: bool, looking_away: bool, distracted: bool
) -> tuple[str, ...]:
    alerts = []
    if tired:
        alerts.append("TIRED")
    if asleep:
        alerts.append("ASLEEP")
    if looking_away:
        alerts.append("LOOKING AWAY")
    if distracted:
        alerts.append("DISTRACTED")
    return tuple(alerts)


class LiveFrameAnalyzer:
    """Turn per-frame observations into time-aware analysis results."""

    def __init__(self, scorer: SupportsAttentionScoring):
        self.scorer = scorer

    def analyze(self, observation: FrameObservation) -> AnalysisResult:
        if not observation.face_detected:
            self.scorer.mark_face_missing(observation.timestamp)
            return AnalysisResult(
                observation=observation,
                state=AnalysisState(),
            )

        tired, perclos = self.scorer.get_rolling_PERCLOS(
            observation.timestamp, observation.ear
        )
        observation = replace(
            observation,
            perclos=perclos,
            perclos_ready=self.scorer.perclos_ready,
        )
        asleep, looking_away, distracted = self.scorer.eval_scores(
            observation.timestamp,
            observation.ear,
            observation.gaze,
            observation.roll,
            observation.pitch,
            observation.yaw,
        )
        state = AnalysisState(
            tired=tired,
            asleep=asleep,
            looking_away=looking_away,
            distracted=distracted,
        )
        return AnalysisResult(
            observation=observation,
            state=state,
            alerts=build_alerts(
                tired=tired,
                asleep=asleep,
                looking_away=looking_away,
                distracted=distracted,
            ),
        )


class LiveMonitoringService:
    """Coordinate live frame observation extraction and analysis."""

    def __init__(
        self,
        frame_source: SupportsObservationSource,
        analyzer: SupportsResultAnalyzer,
        *,
        session_service: SupportsSessionService | None = None,
        observation_sampler: SupportsObservationSampler | None = None,
        baseline_service: SupportsBaselineService | None = None,
    ):
        self.frame_source = frame_source
        self.analyzer = analyzer
        self.session_service = session_service
        self.observation_sampler = observation_sampler
        self.baseline_service = baseline_service

    def stream(self):
        for sample in self.frame_source.stream():
            result = self.analyzer.analyze(sample.observation)
            processing_ms = (time.perf_counter() - sample.processing_started_at) * 1000
            observation = replace(sample.observation, processing_ms=processing_ms)
            result = replace(result, observation=observation)
            result = self._attach_session(result)
            yield sample.frame, result

    def _attach_session(self, result: AnalysisResult) -> AnalysisResult:
        if self.session_service is None:
            return result
        session = self.session_service.get_session()
        if session is None:
            return result
        result = replace(result, session_id=session.session_id)
        if (
            self.session_service.get_status() is SessionStatus.ACTIVE
            and self.observation_sampler is not None
            and self.observation_sampler.should_sample(result.observation.observed_at)
        ):
            self.session_service.register_observation(result)
            if self.baseline_service is not None:
                self.baseline_service.add_observation(result.observation)
        return result
