from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ergosense.application.fatigue_engine import FatigueEngine
from ergosense.application.longitudinal_feature_service import (
    LongitudinalFeatureService,
)
from ergosense.domain.assessment import (
    AssessmentTrend,
    FatigueAssessment,
    WindowAssessmentStatus,
    WindowFatigueAssessment,
)
from ergosense.domain.baseline import BaselineConfidence, PersonalBaseline
from ergosense.domain.fatigue import (
    FatigueEvidence,
    FatigueState,
    FatigueTrend,
    clamp_score,
)
from ergosense.domain.longitudinal import TimeWindow
from ergosense.domain.session import MonitoringSession, SessionObservation


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_windows() -> tuple[TimeWindow, ...]:
    return (
        TimeWindow.one_minute(),
        TimeWindow.five_minutes(),
        TimeWindow.fifteen_minutes(),
    )


@dataclass(frozen=True)
class FatigueAssessmentConfig:
    windows: tuple[TimeWindow, ...] = _default_windows()
    minimum_window_observations: int = 3
    primary_window_confidence_weight: float = 0.5
    intermediate_window_adjustment_weight: float = 0.25
    recent_window_adjustment_weight: float = 0.15
    assessment_trend_score_delta_epsilon: float = 5.0
    assessment_confidence_target_observations: int = 12
    assessment_confidence_target_duration: timedelta = timedelta(minutes=15)
    assessment_confidence_observation_weight: float = 0.25
    assessment_confidence_window_weight: float = 0.25
    assessment_confidence_baseline_weight: float = 0.20
    assessment_confidence_state_weight: float = 0.20
    assessment_confidence_duration_weight: float = 0.10

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("windows cannot be empty")
        if self.minimum_window_observations <= 0:
            raise ValueError("minimum_window_observations must be positive")
        if self.assessment_confidence_target_observations <= 0:
            raise ValueError(
                "assessment_confidence_target_observations must be positive"
            )
        if self.assessment_confidence_target_duration <= timedelta(0):
            raise ValueError("assessment_confidence_target_duration must be positive")
        weights = (
            self.assessment_confidence_observation_weight,
            self.assessment_confidence_window_weight,
            self.assessment_confidence_baseline_weight,
            self.assessment_confidence_state_weight,
            self.assessment_confidence_duration_weight,
        )
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("assessment confidence weights must sum to 1.0")


class FatigueAssessmentService:
    """Coordinate session observations, baseline, features, and fatigue states."""

    def __init__(
        self,
        *,
        feature_service: LongitudinalFeatureService | None = None,
        fatigue_engine: FatigueEngine | None = None,
        config: FatigueAssessmentConfig | None = None,
        session_service: Any | None = None,
        baseline_service: Any | None = None,
        now=utc_now,
    ):
        self.feature_service = feature_service or LongitudinalFeatureService()
        self.fatigue_engine = fatigue_engine or FatigueEngine(now=now)
        self.config = config or FatigueAssessmentConfig()
        self.session_service = session_service
        self.baseline_service = baseline_service
        self._now = now

    def assess(
        self,
        session: MonitoringSession,
        baseline: PersonalBaseline | None,
        observations: tuple[SessionObservation, ...] | list[SessionObservation],
        at: datetime | None = None,
        previous_assessment: FatigueAssessment | None = None,
    ) -> FatigueAssessment:
        if at is not None:
            self._require_aware_datetime(at, "at")
        ordered_observations = tuple(
            sorted(observations, key=lambda item: item.captured_at)
        )
        calculated_at = at or self._resolve_calculated_at(session, ordered_observations)
        session_elapsed = calculated_at - session.started_at
        if session_elapsed < timedelta(0):
            raise ValueError("assessment time cannot be earlier than session start")

        baseline_value = baseline or PersonalBaseline()
        window_states = []
        for window in self.config.windows:
            features = self.feature_service.extract_features(
                ordered_observations,
                baseline_value,
                window,
                at=calculated_at,
            )
            status = self._window_status(
                window, session_elapsed, features.observation_count
            )
            state = None
            if status is WindowAssessmentStatus.AVAILABLE:
                previous_state = self._get_previous_window_state(
                    previous_assessment, window
                )
                state = self.fatigue_engine.evaluate(
                    features, previous_state=previous_state
                )
            window_states.append(
                WindowFatigueAssessment(
                    window=window,
                    status=status,
                    state=state,
                    observation_count=features.observation_count,
                )
            )

        window_states_tuple = tuple(window_states)
        primary_window = self._select_primary_window(window_states_tuple)
        current_state = self._consolidate_current_state(
            session.session_id,
            calculated_at,
            window_states_tuple,
            primary_window,
        )
        assessment_confidence = self._calculate_assessment_confidence(
            baseline,
            window_states_tuple,
            current_state,
            len(ordered_observations),
            session_elapsed,
        )
        trend = self._determine_assessment_trend(
            current_state,
            previous_assessment,
            primary_window,
            window_states_tuple,
        )
        return FatigueAssessment(
            session_id=session.session_id,
            calculated_at=calculated_at,
            current_state=current_state,
            window_states=window_states_tuple,
            primary_window=primary_window,
            available_observations=len(ordered_observations),
            baseline_available=self._is_baseline_ready(baseline),
            session_elapsed=session_elapsed,
            assessment_confidence=assessment_confidence,
            trend=trend,
        )

    def assess_current_session(
        self,
        *,
        at: datetime | None = None,
        previous_assessment: FatigueAssessment | None = None,
    ) -> FatigueAssessment:
        if self.session_service is None:
            raise ValueError("session_service is required for assess_current_session")
        session = self.session_service.get_session()
        if session is None:
            raise ValueError("no session is available")
        baseline = None
        if self.baseline_service is not None:
            baseline = self.baseline_service.get_baseline(at=at)
        observations = self.session_service.list_observations()
        return self.assess(
            session=session,
            baseline=baseline,
            observations=observations,
            at=at,
            previous_assessment=previous_assessment,
        )

    def _window_status(
        self,
        window: TimeWindow,
        session_elapsed: timedelta,
        observation_count: int,
    ) -> WindowAssessmentStatus:
        if session_elapsed < window.duration:
            return WindowAssessmentStatus.INSUFFICIENT_DATA
        if observation_count < self.config.minimum_window_observations:
            return WindowAssessmentStatus.INSUFFICIENT_DATA
        return WindowAssessmentStatus.AVAILABLE

    def _select_primary_window(
        self, window_states: tuple[WindowFatigueAssessment, ...]
    ) -> TimeWindow | None:
        available = [
            item.window
            for item in window_states
            if item.status is WindowAssessmentStatus.AVAILABLE
        ]
        if not available:
            return None
        return max(available, key=lambda window: window.duration)

    def _consolidate_current_state(
        self,
        session_id,
        calculated_at: datetime,
        window_states: tuple[WindowFatigueAssessment, ...],
        primary_window: TimeWindow | None,
    ) -> FatigueState | None:
        if primary_window is None:
            return None
        states_by_name = {
            item.window.name: item.state
            for item in window_states
            if item.status is WindowAssessmentStatus.AVAILABLE
            and item.state is not None
        }
        primary_state = states_by_name.get(primary_window.name)
        if primary_state is None:
            return None
        one_minute_state = states_by_name.get("1m")
        five_minutes_state = states_by_name.get("5m")
        fifteen_minutes_state = states_by_name.get("15m")

        if primary_window.name == "15m":
            score = primary_state.score
            if five_minutes_state is not None:
                score += self.config.intermediate_window_adjustment_weight * (
                    five_minutes_state.score - primary_state.score
                )
            if one_minute_state is not None:
                reference_state = five_minutes_state or primary_state
                score += self.config.recent_window_adjustment_weight * (
                    one_minute_state.score - reference_state.score
                )
            component_weights = self._window_component_weights(
                fifteen_minutes_state, five_minutes_state, one_minute_state
            )
        elif primary_window.name == "5m":
            score = primary_state.score
            if one_minute_state is not None:
                score += self.config.intermediate_window_adjustment_weight * (
                    one_minute_state.score - primary_state.score
                )
            component_weights = self._window_component_weights(
                None, five_minutes_state, one_minute_state
            )
        else:
            score = primary_state.score
            component_weights = self._window_component_weights(
                None, None, one_minute_state
            )

        score = clamp_score(score)
        confidence = self._weighted_confidence(component_weights)
        level = self._level_for_score(score)
        trend = primary_state.trend
        reasons = self._merge_reasons(component_weights)
        return FatigueState(
            session_id=session_id,
            calculated_at=calculated_at,
            score=score,
            level=level,
            trend=trend,
            confidence=confidence,
            ocular_component=self._combine_evidence(
                component_weights, "ocular_component", "ocular"
            ),
            postural_component=self._combine_evidence(
                component_weights, "postural_component", "postural"
            ),
            temporal_component=self._combine_evidence(
                component_weights, "temporal_component", "temporal"
            ),
            convergence_component=self._combine_evidence(
                component_weights, "convergence_component", "convergence"
            ),
            reasons=reasons,
        )

    def _window_component_weights(
        self,
        fifteen_minutes_state: FatigueState | None,
        five_minutes_state: FatigueState | None,
        one_minute_state: FatigueState | None,
    ) -> tuple[tuple[FatigueState, float], ...]:
        weighted_states: list[tuple[FatigueState, float]] = []
        if fifteen_minutes_state is not None:
            weighted_states.append((fifteen_minutes_state, 0.55))
            if five_minutes_state is not None:
                weighted_states.append((five_minutes_state, 0.30))
            if one_minute_state is not None:
                weighted_states.append((one_minute_state, 0.15))
        elif five_minutes_state is not None:
            weighted_states.append((five_minutes_state, 0.70))
            if one_minute_state is not None:
                weighted_states.append((one_minute_state, 0.30))
        elif one_minute_state is not None:
            weighted_states.append((one_minute_state, 1.0))
        return tuple(weighted_states)

    def _combine_evidence(
        self,
        weighted_states: tuple[tuple[FatigueState, float], ...],
        attribute_name: str,
        source: str,
    ) -> FatigueEvidence:
        if not weighted_states:
            return FatigueEvidence(
                source=source,
                score=0.0,
                strength=0.0,
                persistence=0.0,
                explanation="Sin evidencia suficiente en esta dimension.",
            )
        total_weight = sum(weight for _, weight in weighted_states)
        score = (
            sum(
                getattr(state, attribute_name).score * weight
                for state, weight in weighted_states
            )
            / total_weight
        )
        strength = (
            sum(
                getattr(state, attribute_name).strength * weight
                for state, weight in weighted_states
            )
            / total_weight
        )
        persistence = (
            sum(
                getattr(state, attribute_name).persistence * weight
                for state, weight in weighted_states
            )
            / total_weight
        )
        explanation = max(
            (
                getattr(state, attribute_name).explanation
                for state, _ in weighted_states
            ),
            key=len,
            default="Sin evidencia suficiente en esta dimension.",
        )
        return FatigueEvidence(
            source=source,
            score=score,
            strength=strength,
            persistence=persistence,
            explanation=explanation,
        )

    def _weighted_confidence(
        self, weighted_states: tuple[tuple[FatigueState, float], ...]
    ) -> float:
        if not weighted_states:
            return 0.0
        total_weight = sum(weight for _, weight in weighted_states)
        return clamp_score(
            sum(state.confidence * weight for state, weight in weighted_states)
            / total_weight
        )

    def _merge_reasons(
        self, weighted_states: tuple[tuple[FatigueState, float], ...]
    ) -> tuple[str, ...]:
        seen = set()
        merged = []
        for state, _ in weighted_states:
            for reason in state.reasons:
                if reason not in seen:
                    merged.append(reason)
                    seen.add(reason)
        return tuple(merged)

    def _calculate_assessment_confidence(
        self,
        baseline: PersonalBaseline | None,
        window_states: tuple[WindowFatigueAssessment, ...],
        current_state: FatigueState | None,
        available_observations: int,
        session_elapsed: timedelta,
    ) -> float:
        observation_factor = min(
            1.0,
            available_observations
            / self.config.assessment_confidence_target_observations,
        )
        available_window_count = sum(
            item.status is WindowAssessmentStatus.AVAILABLE for item in window_states
        )
        window_factor = available_window_count / len(window_states)
        baseline_factor = self._baseline_factor(baseline)
        state_factor = (
            0.0 if current_state is None else current_state.confidence / 100.0
        )
        duration_factor = min(
            1.0,
            session_elapsed.total_seconds()
            / self.config.assessment_confidence_target_duration.total_seconds(),
        )
        confidence = 100.0 * (
            observation_factor * self.config.assessment_confidence_observation_weight
            + window_factor * self.config.assessment_confidence_window_weight
            + baseline_factor * self.config.assessment_confidence_baseline_weight
            + state_factor * self.config.assessment_confidence_state_weight
            + duration_factor * self.config.assessment_confidence_duration_weight
        )
        return clamp_score(confidence)

    def _determine_assessment_trend(
        self,
        current_state: FatigueState | None,
        previous_assessment: FatigueAssessment | None,
        primary_window: TimeWindow | None,
        window_states: tuple[WindowFatigueAssessment, ...],
    ) -> AssessmentTrend:
        if current_state is None:
            return AssessmentTrend.UNKNOWN
        if (
            previous_assessment is not None
            and previous_assessment.current_state is not None
        ):
            delta = current_state.score - previous_assessment.current_state.score
            if delta > self.config.assessment_trend_score_delta_epsilon:
                return AssessmentTrend.INCREASING
            if delta < -self.config.assessment_trend_score_delta_epsilon:
                return AssessmentTrend.DECREASING
            return AssessmentTrend.STABLE
        if (
            primary_window is None
            or primary_window.duration < TimeWindow.five_minutes().duration
        ):
            return AssessmentTrend.UNKNOWN
        primary_state = self._get_window_state(window_states, primary_window)
        if primary_state is None or primary_state.trend is FatigueTrend.UNKNOWN:
            return AssessmentTrend.UNKNOWN
        return {
            FatigueTrend.STABLE: AssessmentTrend.STABLE,
            FatigueTrend.INCREASING: AssessmentTrend.INCREASING,
            FatigueTrend.DECREASING: AssessmentTrend.DECREASING,
            FatigueTrend.UNKNOWN: AssessmentTrend.UNKNOWN,
        }[primary_state.trend]

    def _get_previous_window_state(
        self, previous_assessment: FatigueAssessment | None, window: TimeWindow
    ) -> FatigueState | None:
        if previous_assessment is None:
            return None
        return self._get_window_state(previous_assessment.window_states, window)

    @staticmethod
    def _get_window_state(
        window_states: tuple[WindowFatigueAssessment, ...],
        window: TimeWindow,
    ) -> FatigueState | None:
        for item in window_states:
            if item.window.name == window.name:
                return item.state
        return None

    def _resolve_calculated_at(
        self, session: MonitoringSession, observations: tuple[SessionObservation, ...]
    ) -> datetime:
        if observations:
            return observations[-1].captured_at
        if session.ended_at is not None:
            return session.ended_at
        return session.started_at

    @staticmethod
    def _is_baseline_ready(baseline: PersonalBaseline | None) -> bool:
        return baseline is not None and baseline.confidence is BaselineConfidence.READY

    @staticmethod
    def _baseline_factor(baseline: PersonalBaseline | None) -> float:
        if baseline is None:
            return 0.0
        if baseline.confidence is BaselineConfidence.READY:
            return 1.0
        if baseline.confidence is BaselineConfidence.CALIBRATING:
            return 0.5
        return 0.0

    @staticmethod
    def _require_aware_datetime(value: datetime, field_name: str) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware")

    def _level_for_score(self, score: float):
        engine_config = getattr(self.fatigue_engine, "config", None)
        if engine_config is None:
            engine_config = FatigueEngine().config
        return engine_config.level_for_score(score)
