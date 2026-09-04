from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True)
class FrameObservation:
    """Metrics observed for a single instant in the live pipeline."""

    timestamp: float
    observed_at: datetime
    face_detected: bool
    ear: float | None = None
    gaze: float | None = None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    fps: float | None = None
    processing_ms: float | None = None

    def __post_init__(self) -> None:
        _require_aware_datetime(self.observed_at, "observed_at")


@dataclass(frozen=True)
class AnalysisState:
    """Current interpreted states derived from observed metrics."""

    tired: bool = False
    asleep: bool = False
    looking_away: bool = False
    distracted: bool = False


@dataclass(frozen=True)
class AnalysisResult:
    """Analysis outcome for a single observed instant."""

    observation: FrameObservation
    state: AnalysisState
    session_id: UUID | None = None
    alerts: tuple[str, ...] = ()
    perclos: float | None = None
    perclos_ready: bool = False
