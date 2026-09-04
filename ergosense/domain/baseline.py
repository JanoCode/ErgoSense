from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID


def _empty_metric() -> "MetricBaseline":
    return MetricBaseline()


class BaselineConfidence(Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    CALIBRATING = "calibrating"
    READY = "ready"


@dataclass(frozen=True)
class MetricBaseline:
    mean: float = 0.0
    standard_deviation: float = 0.0
    sample_count: int = 0


@dataclass(frozen=True)
class EyeBaseline:
    ear: MetricBaseline = field(default_factory=_empty_metric)
    perclos: MetricBaseline = field(default_factory=_empty_metric)


@dataclass(frozen=True)
class HeadBaseline:
    pitch: MetricBaseline = field(default_factory=_empty_metric)
    yaw: MetricBaseline = field(default_factory=_empty_metric)
    roll: MetricBaseline = field(default_factory=_empty_metric)


@dataclass(frozen=True)
class PersonalBaseline:
    session_id: UUID | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    calibration_duration: timedelta = timedelta(0)
    confidence: BaselineConfidence = BaselineConfidence.INSUFFICIENT_DATA
    eye: EyeBaseline = field(default_factory=EyeBaseline)
    head: HeadBaseline = field(default_factory=HeadBaseline)
