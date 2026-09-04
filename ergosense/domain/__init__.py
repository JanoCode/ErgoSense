"""Domain models for ErgoSense."""

from ergosense.domain.assessment import (
    AssessmentTrend,
    FatigueAssessment,
    WindowAssessmentStatus,
    WindowFatigueAssessment,
)
from ergosense.domain.baseline import (
    BaselineConfidence,
    EyeBaseline,
    HeadBaseline,
    MetricBaseline,
    PersonalBaseline,
)
from ergosense.domain.fatigue import (
    FatigueEngineConfig,
    FatigueEvidence,
    FatigueLevel,
    FatigueLevelThresholds,
    FatigueState,
    FatigueTrend,
    clamp_score,
)
from ergosense.domain.fatigue_timeline import (
    FatigueTimeline,
    FatigueTimelineConfig,
    FatigueTimelinePoint,
    FatigueTransition,
)
from ergosense.domain.longitudinal import (
    LongitudinalFeatures,
    MetricAggregation,
    MetricDeviation,
    MetricTrend,
    ProlongedEvent,
    TimeWindow,
    TrendDirection,
)
from ergosense.domain.monitoring import AnalysisResult, AnalysisState, FrameObservation
from ergosense.domain.session import (
    MonitoringSession,
    SessionObservation,
    SessionStatus,
)

__all__ = [
    "AnalysisResult",
    "AnalysisState",
    "AssessmentTrend",
    "BaselineConfidence",
    "EyeBaseline",
    "FatigueAssessment",
    "FatigueEngineConfig",
    "FatigueEvidence",
    "FatigueLevel",
    "FatigueLevelThresholds",
    "FatigueState",
    "FatigueTrend",
    "FatigueTimeline",
    "FatigueTimelineConfig",
    "FatigueTimelinePoint",
    "FatigueTransition",
    "FrameObservation",
    "HeadBaseline",
    "LongitudinalFeatures",
    "MetricBaseline",
    "MetricAggregation",
    "MetricDeviation",
    "MetricTrend",
    "MonitoringSession",
    "PersonalBaseline",
    "ProlongedEvent",
    "SessionObservation",
    "SessionStatus",
    "TimeWindow",
    "TrendDirection",
    "WindowAssessmentStatus",
    "WindowFatigueAssessment",
    "clamp_score",
]
