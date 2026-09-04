from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from ergosense.application.fatigue_assessment_service import FatigueAssessmentService
from ergosense.application.fatigue_timeline_service import FatigueTimelineService
from ergosense.application.session_service import SessionService
from ergosense.domain.assessment import FatigueAssessment
from ergosense.domain.fatigue_timeline import FatigueTimeline
from ergosense.domain.session import MonitoringSession, SessionObservation


@dataclass(frozen=True)
class SessionFatigueMonitorConfig:
    assessment_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.assessment_interval_seconds <= 0:
            raise ValueError("assessment_interval_seconds must be positive")


class SessionFatigueMonitor:
    """Coordinate periodic fatigue assessment generation for an active session."""

    def __init__(
        self,
        *,
        session_service: SessionService,
        assessment_service: FatigueAssessmentService,
        timeline_service: FatigueTimelineService | None = None,
        config: SessionFatigueMonitorConfig | None = None,
    ):
        self.session_service = session_service
        self.assessment_service = assessment_service
        self.timeline_service = timeline_service or FatigueTimelineService()
        self.config = config or SessionFatigueMonitorConfig()
        self._running = False
        self._timeline: FatigueTimeline | None = None
        self._latest_assessment: FatigueAssessment | None = None
        self._last_assessment_at = None

    def start_session(
        self,
        *,
        started_at=None,
        session_id=None,
    ) -> MonitoringSession:
        if self._running:
            raise ValueError("monitor is already running")
        session = self.session_service.start_session(
            started_at=started_at,
            session_id=session_id,
        )
        self._timeline = self.timeline_service.create_timeline(session.session_id)
        baseline_service = getattr(self.assessment_service, "baseline_service", None)
        if baseline_service is not None:
            baseline_service.start_calibration(session=session)
        self._latest_assessment = None
        self._last_assessment_at = None
        self._running = True
        return session

    def process_observation(
        self, observation: SessionObservation
    ) -> FatigueAssessment | None:
        if not self._running:
            raise ValueError("monitor is not running")
        session = self.session_service.require_active_session()
        if observation.session_id != session.session_id:
            raise ValueError("observation session_id does not match active session")
        self.session_service.add_session_observation(observation)
        baseline_service = getattr(self.assessment_service, "baseline_service", None)
        if baseline_service is not None:
            baseline_service.add_observation(observation.result.observation)
        if not self._should_assess(observation, session):
            return None
        assessment = self.assessment_service.assess_current_session(
            at=observation.captured_at,
            previous_assessment=self._latest_assessment,
        )
        self._timeline = self.timeline_service.add_assessment(
            self._require_timeline(), assessment
        )
        self._latest_assessment = assessment
        self._last_assessment_at = assessment.calculated_at
        return assessment

    def current_assessment(self) -> FatigueAssessment | None:
        return self._latest_assessment

    def latest_assessment(self) -> FatigueAssessment | None:
        return self._latest_assessment

    def timeline(self) -> FatigueTimeline | None:
        return self._timeline

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> FatigueTimeline | None:
        self._running = False
        return self._timeline

    def _should_assess(
        self, observation: SessionObservation, session: MonitoringSession
    ) -> bool:
        interval = timedelta(seconds=self.config.assessment_interval_seconds)
        if self._last_assessment_at is None:
            return observation.captured_at - session.started_at >= interval
        return observation.captured_at - self._last_assessment_at >= interval

    def _require_timeline(self) -> FatigueTimeline:
        if self._timeline is None:
            raise ValueError("timeline has not been initialized")
        return self._timeline
