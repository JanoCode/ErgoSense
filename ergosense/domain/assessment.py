from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ergosense.domain.fatigue import FatigueState, clamp_score
from ergosense.domain.longitudinal import TimeWindow


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class WindowAssessmentStatus(Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    AVAILABLE = "available"


class AssessmentTrend(Enum):
    STABLE = "stable"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WindowFatigueAssessment:
    window: TimeWindow
    status: WindowAssessmentStatus
    state: FatigueState | None
    observation_count: int


@dataclass(frozen=True)
class FatigueAssessment:
    session_id: object
    calculated_at: datetime
    current_state: FatigueState | None
    window_states: tuple[WindowFatigueAssessment, ...]
    primary_window: TimeWindow | None
    available_observations: int
    baseline_available: bool
    session_elapsed: timedelta | None
    assessment_confidence: float
    trend: AssessmentTrend

    def __post_init__(self) -> None:
        _require_aware_datetime(self.calculated_at, "calculated_at")
        object.__setattr__(
            self, "assessment_confidence", clamp_score(self.assessment_confidence)
        )
