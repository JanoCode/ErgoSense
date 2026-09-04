from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ergosense.domain.assessment import AssessmentTrend, FatigueAssessment
from ergosense.domain.fatigue import FatigueLevel


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True)
class FatigueTimelineConfig:
    sustained_fatigue_points: int = 2
    trend_score_tolerance: float = 5.0

    def __post_init__(self) -> None:
        if self.sustained_fatigue_points <= 0:
            raise ValueError("sustained_fatigue_points must be positive")
        if self.trend_score_tolerance < 0:
            raise ValueError("trend_score_tolerance cannot be negative")


@dataclass(frozen=True)
class FatigueTimelinePoint:
    assessment: FatigueAssessment
    session_elapsed: timedelta

    def __post_init__(self) -> None:
        _require_aware_datetime(
            self.assessment.calculated_at, "assessment.calculated_at"
        )
        if self.session_elapsed < timedelta(0):
            raise ValueError("session_elapsed cannot be negative")

    @property
    def calculated_at(self) -> datetime:
        return self.assessment.calculated_at

    @property
    def state(self):
        return self.assessment.current_state

    @property
    def score(self) -> float | None:
        if self.assessment.current_state is None:
            return None
        return self.assessment.current_state.score

    @property
    def confidence(self) -> float | None:
        if self.assessment.current_state is None:
            return None
        return self.assessment.current_state.confidence

    @property
    def trend(self):
        if self.assessment.current_state is None:
            return self.assessment.trend
        return self.assessment.current_state.trend


@dataclass(frozen=True)
class FatigueTransition:
    from_level: FatigueLevel
    to_level: FatigueLevel
    occurred_at: datetime
    session_elapsed: timedelta
    score_before: float
    score_after: float

    def __post_init__(self) -> None:
        _require_aware_datetime(self.occurred_at, "occurred_at")
        if self.session_elapsed < timedelta(0):
            raise ValueError("session_elapsed cannot be negative")


@dataclass(frozen=True)
class FatigueTimeline:
    session_id: object
    points: tuple[FatigueTimelinePoint, ...] = field(default_factory=tuple)

    def add(self, assessment: FatigueAssessment) -> "FatigueTimeline":
        if assessment.session_id != self.session_id:
            raise ValueError("assessment session_id does not match timeline session_id")
        if self.points:
            latest = self.points[-1]
            if assessment.calculated_at < latest.calculated_at:
                raise ValueError(
                    "assessment timestamp cannot be earlier than latest point"
                )
            if assessment.calculated_at == latest.calculated_at:
                raise ValueError("duplicate assessment timestamp is not allowed")
        point = FatigueTimelinePoint(
            assessment=assessment,
            session_elapsed=assessment.session_elapsed or timedelta(0),
        )
        return FatigueTimeline(
            session_id=self.session_id, points=self.points + (point,)
        )

    @property
    def latest(self) -> FatigueTimelinePoint | None:
        if not self.points:
            return None
        return self.points[-1]

    @property
    def first(self) -> FatigueTimelinePoint | None:
        if not self.points:
            return None
        return self.points[0]

    @property
    def is_empty(self) -> bool:
        return not self.points

    @property
    def duration(self) -> timedelta:
        if len(self.points) < 2:
            return timedelta(0)
        return self.points[-1].calculated_at - self.points[0].calculated_at

    @property
    def peak_score(self) -> float | None:
        scores = [point.score for point in self.points if point.score is not None]
        if not scores:
            return None
        return max(scores)

    def time_to_first_fatigue(self) -> timedelta | None:
        first = self.first
        if first is None:
            return None
        for point in self.points:
            if point.state is not None and point.state.level is not FatigueLevel.NORMAL:
                return point.calculated_at - first.calculated_at
        return None

    def first_sustained_fatigue(
        self, config: FatigueTimelineConfig | None = None
    ) -> FatigueTimelinePoint | None:
        config = config or FatigueTimelineConfig()
        consecutive = 0
        for point in self.points:
            if point.state is not None and point.state.level is not FatigueLevel.NORMAL:
                consecutive += 1
                if consecutive >= config.sustained_fatigue_points:
                    return point
            else:
                consecutive = 0
        return None

    def transitions(self) -> tuple[FatigueTransition, ...]:
        transitions = []
        previous_point = None
        for point in self.points:
            if point.state is None:
                previous_point = point
                continue
            if previous_point is None or previous_point.state is None:
                previous_point = point
                continue
            if previous_point.state.level is not point.state.level:
                transitions.append(
                    FatigueTransition(
                        from_level=previous_point.state.level,
                        to_level=point.state.level,
                        occurred_at=point.calculated_at,
                        session_elapsed=point.session_elapsed,
                        score_before=previous_point.state.score,
                        score_after=point.state.score,
                    )
                )
            previous_point = point
        return tuple(transitions)

    def trend(self, config: FatigueTimelineConfig | None = None) -> AssessmentTrend:
        config = config or FatigueTimelineConfig()
        scored_points = [point for point in self.points if point.score is not None]
        if len(scored_points) < 2:
            return AssessmentTrend.UNKNOWN
        delta = scored_points[-1].score - scored_points[0].score
        if delta > config.trend_score_tolerance:
            return AssessmentTrend.INCREASING
        if delta < -config.trend_score_tolerance:
            return AssessmentTrend.DECREASING
        return AssessmentTrend.STABLE
