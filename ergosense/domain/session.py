from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID

from ergosense.domain.monitoring import AnalysisResult


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class SessionStatus(Enum):
    ACTIVE = "active"
    ENDED = "ended"


@dataclass(frozen=True)
class MonitoringSession:
    """A live monitoring session for a continuous work period."""

    session_id: UUID
    started_at: datetime
    status: SessionStatus
    ended_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_aware_datetime(self.started_at, "started_at")
        if self.ended_at is not None:
            _require_aware_datetime(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at cannot be earlier than started_at")


@dataclass(frozen=True)
class SessionObservation:
    """A sampled analysis result associated with a monitoring session."""

    session_id: UUID
    captured_at: datetime
    elapsed_since_start: timedelta
    result: AnalysisResult

    def __post_init__(self) -> None:
        _require_aware_datetime(self.captured_at, "captured_at")
        if self.elapsed_since_start < timedelta(0):
            raise ValueError("elapsed_since_start cannot be negative")
