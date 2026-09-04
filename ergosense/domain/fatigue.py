from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Callable
from uuid import UUID


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def clamp_score(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(100.0, float(value)))


class FatigueLevel(Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FatigueTrend(Enum):
    STABLE = "stable"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FatigueEvidence:
    source: str
    score: float
    strength: float
    persistence: float
    explanation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", clamp_score(self.score))
        object.__setattr__(self, "strength", clamp_score(self.strength))
        object.__setattr__(self, "persistence", clamp_score(self.persistence))


@dataclass(frozen=True)
class FatigueLevelThresholds:
    mild: float = 20.0
    moderate: float = 40.0
    high: float = 60.0
    very_high: float = 80.0

    def __post_init__(self) -> None:
        values = (self.mild, self.moderate, self.high, self.very_high)
        if any(not isfinite(value) for value in values):
            raise ValueError("level thresholds must be finite")
        if not (0 <= self.mild <= self.moderate <= self.high <= self.very_high <= 100):
            raise ValueError("level thresholds must be ordered within 0..100")


@dataclass(frozen=True)
class FatigueEngineConfig:
    level_thresholds: FatigueLevelThresholds = field(
        default_factory=FatigueLevelThresholds
    )
    ocular_weight: float = 0.40
    postural_weight: float = 0.25
    temporal_weight: float = 0.15
    convergence_weight: float = 0.20
    ocular_z_score_reference: float = 2.5
    ocular_perclos_difference_reference: float = 0.12
    ocular_ear_difference_reference: float = 0.05
    ocular_persistence_target_samples: int = 6
    ocular_trend_bonus: float = 12.0
    postural_z_score_reference: float = 2.5
    postural_difference_reference_degrees: float = 12.0
    postural_persistence_target_samples: int = 6
    postural_trend_bonus: float = 8.0
    temporal_exposure_start: timedelta = timedelta(minutes=30)
    temporal_exposure_full: timedelta = timedelta(hours=4)
    temporal_max_component_score: float = 40.0
    convergence_signal_threshold: float = 25.0
    convergence_bonus_two_signals: float = 45.0
    convergence_bonus_three_signals: float = 75.0
    convergence_bonus_all_signals: float = 100.0
    confidence_min_observations: int = 4
    confidence_target_duration: timedelta = timedelta(minutes=10)
    confidence_sample_weight: float = 0.35
    confidence_baseline_weight: float = 0.30
    confidence_metric_weight: float = 0.20
    confidence_duration_weight: float = 0.15
    trend_score_delta_epsilon: float = 7.0
    stabilization_alpha: float = 0.35
    stabilization_max_delta: float = 18.0

    def __post_init__(self) -> None:
        weights = (
            self.ocular_weight,
            self.postural_weight,
            self.temporal_weight,
            self.convergence_weight,
        )
        if any(not isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("component weights must be finite and non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("component weights must sum to 1.0")
        confidence_weights = (
            self.confidence_sample_weight,
            self.confidence_baseline_weight,
            self.confidence_metric_weight,
            self.confidence_duration_weight,
        )
        if any(not isfinite(weight) or weight < 0 for weight in confidence_weights):
            raise ValueError("confidence weights must be finite and non-negative")
        if abs(sum(confidence_weights) - 1.0) > 1e-9:
            raise ValueError("confidence weights must sum to 1.0")
        if self.ocular_persistence_target_samples <= 0:
            raise ValueError("ocular_persistence_target_samples must be positive")
        if self.postural_persistence_target_samples <= 0:
            raise ValueError("postural_persistence_target_samples must be positive")
        if self.confidence_min_observations <= 0:
            raise ValueError("confidence_min_observations must be positive")
        if self.temporal_exposure_start < timedelta(0):
            raise ValueError("temporal_exposure_start cannot be negative")
        if self.temporal_exposure_full <= self.temporal_exposure_start:
            raise ValueError(
                "temporal_exposure_full must be greater than temporal_exposure_start"
            )
        if self.confidence_target_duration <= timedelta(0):
            raise ValueError("confidence_target_duration must be positive")
        if self.stabilization_alpha < 0 or self.stabilization_alpha > 1:
            raise ValueError("stabilization_alpha must be within 0..1")
        if self.stabilization_max_delta < 0:
            raise ValueError("stabilization_max_delta cannot be negative")
        if self.trend_score_delta_epsilon < 0:
            raise ValueError("trend_score_delta_epsilon cannot be negative")

    def level_for_score(self, score: float) -> FatigueLevel:
        score = clamp_score(score)
        if score >= self.level_thresholds.very_high:
            return FatigueLevel.VERY_HIGH
        if score >= self.level_thresholds.high:
            return FatigueLevel.HIGH
        if score >= self.level_thresholds.moderate:
            return FatigueLevel.MODERATE
        if score >= self.level_thresholds.mild:
            return FatigueLevel.MILD
        return FatigueLevel.NORMAL


@dataclass(frozen=True)
class FatigueState:
    session_id: UUID | None
    calculated_at: datetime
    score: float
    level: FatigueLevel
    trend: FatigueTrend
    confidence: float
    ocular_component: FatigueEvidence
    postural_component: FatigueEvidence
    temporal_component: FatigueEvidence
    convergence_component: FatigueEvidence
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_aware_datetime(self.calculated_at, "calculated_at")
        object.__setattr__(self, "score", clamp_score(self.score))
        object.__setattr__(self, "confidence", clamp_score(self.confidence))
