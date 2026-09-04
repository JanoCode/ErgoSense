from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import UUID, uuid4

from ergosense.domain.monitoring import AnalysisResult
from ergosense.domain.session import (
    MonitoringSession,
    SessionObservation,
    SessionStatus,
)

DEFAULT_OBSERVATION_SAMPLE_INTERVAL = timedelta(seconds=5)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ObservationSampler:
    """Allow longitudinal observation registration at a lower fixed cadence."""

    def __init__(
        self,
        interval: timedelta = DEFAULT_OBSERVATION_SAMPLE_INTERVAL,
    ):
        if interval <= timedelta(0):
            raise ValueError("sampling interval must be positive")
        self.interval = interval
        self._next_sample_at: datetime | None = None

    def start(self, started_at: datetime) -> None:
        self._next_sample_at = started_at + self.interval

    def should_sample(self, observed_at: datetime) -> bool:
        if self._next_sample_at is None:
            return False
        if observed_at < self._next_sample_at:
            return False
        while observed_at >= self._next_sample_at:
            self._next_sample_at += self.interval
        return True


class SessionService:
    """Manage the lifecycle and in-memory observations of one session."""

    def __init__(self, now: Callable[[], datetime] = utc_now):
        self._now = now
        self._session: MonitoringSession | None = None
        self._observations: list[SessionObservation] = []

    def start_session(
        self, *, started_at: datetime | None = None, session_id: UUID | None = None
    ) -> MonitoringSession:
        if self._session is not None and self._session.status is SessionStatus.ACTIVE:
            raise ValueError("cannot start a new session while another is active")
        started = started_at or self._now()
        session = MonitoringSession(
            session_id=session_id or uuid4(),
            started_at=started,
            status=SessionStatus.ACTIVE,
        )
        self._session = session
        self._observations = []
        return session

    def end_session(self, *, ended_at: datetime | None = None) -> MonitoringSession:
        session = self.require_active_session()
        ended = ended_at or self._now()
        if ended < session.started_at:
            raise ValueError("session end cannot be earlier than its start")
        self._session = replace(
            session,
            ended_at=ended,
            status=SessionStatus.ENDED,
        )
        return self._session

    def get_session(self) -> MonitoringSession | None:
        return self._session

    def require_active_session(self) -> MonitoringSession:
        if self._session is None or self._session.status is not SessionStatus.ACTIVE:
            raise ValueError("no active session")
        return self._session

    def get_status(self) -> SessionStatus | None:
        if self._session is None:
            return None
        return self._session.status

    def get_duration(self, *, at: datetime | None = None) -> timedelta:
        session = self._session
        if session is None:
            raise ValueError("no session has been started")
        end_time = (
            session.ended_at if session.ended_at is not None else (at or self._now())
        )
        if end_time < session.started_at:
            raise ValueError("duration cannot be negative")
        return end_time - session.started_at

    def register_observation(self, result: AnalysisResult) -> SessionObservation:
        session = self.require_active_session()
        captured_at = result.observation.observed_at
        if captured_at < session.started_at:
            raise ValueError("observation cannot be earlier than session start")
        session_result = replace(result, session_id=session.session_id)
        observation = SessionObservation(
            session_id=session.session_id,
            captured_at=captured_at,
            elapsed_since_start=captured_at - session.started_at,
            result=session_result,
        )
        self._observations.append(observation)
        return observation

    def list_observations(self) -> tuple[SessionObservation, ...]:
        return tuple(self._observations)
