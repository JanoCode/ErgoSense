import math
from datetime import datetime, timezone

from ergosense.domain.fatigue import (
    FatigueEngineConfig,
    FatigueEvidence,
    FatigueLevel,
    FatigueState,
    FatigueTrend,
    clamp_score,
)
from ergosense.domain.longitudinal import (
    LongitudinalFeatures,
    MetricAggregation,
    MetricDeviation,
    MetricTrend,
    TrendDirection,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FatigueEngine:
    """Estimate non-clinical fatigue evidence from longitudinal features."""

    def __init__(
        self,
        *,
        config: FatigueEngineConfig | None = None,
        now=utc_now,
    ):
        self.config = config or FatigueEngineConfig()
        self._now = now

    def evaluate(
        self,
        features: LongitudinalFeatures,
        previous_state: FatigueState | None = None,
    ) -> FatigueState:
        ocular_component = self._evaluate_ocular_component(features)
        postural_component = self._evaluate_postural_component(features)
        temporal_component = self._evaluate_temporal_component(features)
        convergence_component = self._evaluate_convergence_component(
            ocular_component, postural_component, temporal_component
        )

        raw_score = (
            ocular_component.score * self.config.ocular_weight
            + postural_component.score * self.config.postural_weight
            + temporal_component.score * self.config.temporal_weight
            + convergence_component.score * self.config.convergence_weight
        )
        score = self._stabilize_score(raw_score, previous_state)
        level = self.config.level_for_score(score)
        confidence = self._calculate_confidence(features)
        trend = self._determine_trend(
            score,
            features,
            previous_state,
            ocular_component,
            postural_component,
        )
        calculated_at = features.window_ended_at or self._now()
        reasons = self._build_reasons(
            ocular_component,
            postural_component,
            temporal_component,
            convergence_component,
        )
        return FatigueState(
            session_id=features.session_id,
            calculated_at=calculated_at,
            score=score,
            level=level,
            trend=trend,
            confidence=confidence,
            ocular_component=ocular_component,
            postural_component=postural_component,
            temporal_component=temporal_component,
            convergence_component=convergence_component,
            reasons=reasons,
        )

    def _evaluate_ocular_component(
        self, features: LongitudinalFeatures
    ) -> FatigueEvidence:
        perclos_strength = self._mean(
            self._positive_difference_score(
                features.perclos_deviation,
                reference=self.config.ocular_perclos_difference_reference,
            ),
            self._positive_z_score(
                features.perclos_deviation, self.config.ocular_z_score_reference
            ),
        )
        ear_strength = self._mean(
            self._negative_difference_score(
                features.ear_deviation,
                reference=self.config.ocular_ear_difference_reference,
            ),
            self._negative_z_score(
                features.ear_deviation, self.config.ocular_z_score_reference
            ),
        )
        strength = clamp_score((perclos_strength * 0.65) + (ear_strength * 0.35))
        persistence = self._mean(
            self._persistence_from_count(
                features.perclos_aggregation.sample_count,
                self.config.ocular_persistence_target_samples,
            ),
            self._persistence_from_count(
                features.ear_aggregation.sample_count,
                self.config.ocular_persistence_target_samples,
            ),
        )
        trend_bonus = 0.0
        if features.perclos_trend.direction is TrendDirection.INCREASING:
            trend_bonus += self.config.ocular_trend_bonus * 0.6
        if features.ear_trend.direction is TrendDirection.DECREASING:
            trend_bonus += self.config.ocular_trend_bonus * 0.4
        score = clamp_score(
            strength * (0.5 + (0.5 * persistence / 100.0)) + trend_bonus
        )
        return FatigueEvidence(
            source="ocular",
            score=score,
            strength=strength,
            persistence=persistence,
            explanation=self._build_ocular_explanation(features, score),
        )

    def _evaluate_postural_component(
        self, features: LongitudinalFeatures
    ) -> FatigueEvidence:
        axis_strengths = [
            self._axis_postural_strength(
                features.pitch_deviation, features.pitch_aggregation
            ),
            self._axis_postural_strength(
                features.yaw_deviation, features.yaw_aggregation
            ),
            self._axis_postural_strength(
                features.roll_deviation, features.roll_aggregation
            ),
        ]
        valid_strengths = [
            strength for strength in axis_strengths if strength is not None
        ]
        strength = self._mean(*valid_strengths)
        persistence = self._mean(
            self._persistence_from_count(
                features.pitch_aggregation.sample_count,
                self.config.postural_persistence_target_samples,
            ),
            self._persistence_from_count(
                features.yaw_aggregation.sample_count,
                self.config.postural_persistence_target_samples,
            ),
            self._persistence_from_count(
                features.roll_aggregation.sample_count,
                self.config.postural_persistence_target_samples,
            ),
        )
        trend_bonus = 0.0
        if self._postural_trend_supports_change(features.pitch_trend):
            trend_bonus += self.config.postural_trend_bonus / 3
        if self._postural_trend_supports_change(features.yaw_trend):
            trend_bonus += self.config.postural_trend_bonus / 3
        if self._postural_trend_supports_change(features.roll_trend):
            trend_bonus += self.config.postural_trend_bonus / 3
        score = clamp_score(
            strength * (0.4 + (0.6 * persistence / 100.0)) + trend_bonus
        )
        return FatigueEvidence(
            source="postural",
            score=score,
            strength=strength,
            persistence=persistence,
            explanation=self._build_postural_explanation(features, score),
        )

    def _evaluate_temporal_component(
        self, features: LongitudinalFeatures
    ) -> FatigueEvidence:
        exposure = features.continuous_time_since_session_start
        if exposure is None:
            return FatigueEvidence(
                source="temporal",
                score=0.0,
                strength=0.0,
                persistence=0.0,
                explanation="Sin suficiente contexto temporal de sesion.",
            )
        if exposure <= self.config.temporal_exposure_start:
            normalized = 0.0
        else:
            span = (
                self.config.temporal_exposure_full - self.config.temporal_exposure_start
            ).total_seconds()
            if span <= 0:
                normalized = 1.0
            else:
                normalized = min(
                    1.0,
                    max(
                        0.0,
                        (exposure - self.config.temporal_exposure_start).total_seconds()
                        / span,
                    ),
                )
        score = clamp_score(normalized * self.config.temporal_max_component_score)
        return FatigueEvidence(
            source="temporal",
            score=score,
            strength=score,
            persistence=score,
            explanation=self._build_temporal_explanation(exposure, score),
        )

    def _evaluate_convergence_component(
        self,
        ocular_component: FatigueEvidence,
        postural_component: FatigueEvidence,
        temporal_component: FatigueEvidence,
    ) -> FatigueEvidence:
        components = (ocular_component, postural_component, temporal_component)
        active_count = sum(
            component.score >= self.config.convergence_signal_threshold
            for component in components
        )
        if active_count >= 3:
            score = self.config.convergence_bonus_all_signals
            explanation = "Coincidencia de senales oculares, posturales y temporales."
        elif active_count == 2:
            score = self.config.convergence_bonus_two_signals
            if (
                ocular_component.score >= self.config.convergence_signal_threshold
                and postural_component.score >= self.config.convergence_signal_threshold
            ):
                explanation = "Coincidencia de senales oculares y posturales."
            elif (
                ocular_component.score >= self.config.convergence_signal_threshold
                and temporal_component.score >= self.config.convergence_signal_threshold
            ):
                explanation = (
                    "Coincidencia de senales oculares con exposicion prolongada."
                )
            else:
                explanation = (
                    "Coincidencia de senales posturales con exposicion prolongada."
                )
        elif active_count == 1:
            score = 10.0
            explanation = "Solo una dimension aporta evidencia relevante."
        else:
            score = 0.0
            explanation = (
                "No hay convergencia suficiente entre dimensiones independientes."
            )
        strength = clamp_score(max(component.score for component in components))
        persistence = clamp_score(
            self._mean(
                ocular_component.persistence,
                postural_component.persistence,
                temporal_component.persistence,
            )
        )
        return FatigueEvidence(
            source="convergence",
            score=score,
            strength=strength,
            persistence=persistence,
            explanation=explanation,
        )

    def _calculate_confidence(self, features: LongitudinalFeatures) -> float:
        sample_factor = min(
            1.0,
            features.observation_count / self.config.confidence_min_observations,
        )
        baseline_available_count = sum(
            deviation.baseline_available
            for deviation in (
                features.ear_deviation,
                features.perclos_deviation,
                features.pitch_deviation,
                features.yaw_deviation,
                features.roll_deviation,
            )
        )
        baseline_factor = baseline_available_count / 5.0
        metric_available_count = sum(
            deviation.current is not None
            for deviation in (
                features.ear_deviation,
                features.perclos_deviation,
                features.pitch_deviation,
                features.yaw_deviation,
                features.roll_deviation,
            )
        )
        metric_factor = metric_available_count / 5.0
        duration = features.continuous_time_since_session_start
        if duration is None:
            duration_factor = 0.0
        else:
            duration_factor = min(
                1.0,
                duration.total_seconds()
                / self.config.confidence_target_duration.total_seconds(),
            )
        confidence = 100.0 * (
            sample_factor * self.config.confidence_sample_weight
            + baseline_factor * self.config.confidence_baseline_weight
            + metric_factor * self.config.confidence_metric_weight
            + duration_factor * self.config.confidence_duration_weight
        )
        return clamp_score(confidence)

    def _determine_trend(
        self,
        score: float,
        features: LongitudinalFeatures,
        previous_state: FatigueState | None,
        ocular_component: FatigueEvidence,
        postural_component: FatigueEvidence,
    ) -> FatigueTrend:
        if previous_state is not None:
            delta = score - previous_state.score
            if delta > self.config.trend_score_delta_epsilon:
                return FatigueTrend.INCREASING
            if delta < -self.config.trend_score_delta_epsilon:
                return FatigueTrend.DECREASING
            return FatigueTrend.STABLE

        directions = [
            features.perclos_trend.direction,
            features.ear_trend.direction,
            features.pitch_trend.direction,
            features.yaw_trend.direction,
            features.roll_trend.direction,
        ]
        known_directions = [
            direction for direction in directions if direction is not None
        ]
        if not known_directions:
            return FatigueTrend.UNKNOWN
        if ocular_component.score >= self.config.convergence_signal_threshold and (
            features.perclos_trend.direction is TrendDirection.INCREASING
            or features.ear_trend.direction is TrendDirection.DECREASING
        ):
            return FatigueTrend.INCREASING
        if postural_component.score >= self.config.convergence_signal_threshold and any(
            trend.direction in {TrendDirection.INCREASING, TrendDirection.DECREASING}
            for trend in (features.pitch_trend, features.yaw_trend, features.roll_trend)
        ):
            return FatigueTrend.INCREASING
        if all(direction is TrendDirection.STABLE for direction in known_directions):
            return FatigueTrend.STABLE
        return FatigueTrend.UNKNOWN

    def _build_reasons(
        self,
        ocular_component: FatigueEvidence,
        postural_component: FatigueEvidence,
        temporal_component: FatigueEvidence,
        convergence_component: FatigueEvidence,
    ) -> tuple[str, ...]:
        reasons = []
        for component in (
            ocular_component,
            postural_component,
            temporal_component,
            convergence_component,
        ):
            if component.score >= 15:
                reasons.append(component.explanation)
        seen = set()
        ordered = []
        for reason in reasons:
            if reason not in seen:
                ordered.append(reason)
                seen.add(reason)
        return tuple(ordered)

    def _stabilize_score(
        self, raw_score: float, previous_state: FatigueState | None
    ) -> float:
        raw_score = clamp_score(raw_score)
        if previous_state is None:
            return raw_score
        blended = (
            previous_state.score * (1.0 - self.config.stabilization_alpha)
            + raw_score * self.config.stabilization_alpha
        )
        delta = blended - previous_state.score
        max_delta = self.config.stabilization_max_delta
        if delta > max_delta:
            blended = previous_state.score + max_delta
        elif delta < -max_delta:
            blended = previous_state.score - max_delta
        return clamp_score(blended)

    def _axis_postural_strength(
        self,
        deviation: MetricDeviation,
        aggregation: MetricAggregation,
    ) -> float | None:
        signals = []
        if deviation.absolute_difference is not None:
            signals.append(
                self._normalize_positive(
                    abs(deviation.absolute_difference),
                    self.config.postural_difference_reference_degrees,
                )
            )
        if deviation.z_score is not None:
            signals.append(
                self._normalize_positive(
                    abs(deviation.z_score), self.config.postural_z_score_reference
                )
            )
        if aggregation.standard_deviation is not None and math.isfinite(
            aggregation.standard_deviation
        ):
            signals.append(
                self._normalize_positive(
                    aggregation.standard_deviation,
                    self.config.postural_difference_reference_degrees,
                )
            )
        if not signals:
            return None
        return self._mean(*signals)

    @staticmethod
    def _postural_trend_supports_change(trend: MetricTrend) -> bool:
        return trend.direction in {
            TrendDirection.INCREASING,
            TrendDirection.DECREASING,
        }

    def _build_ocular_explanation(
        self, features: LongitudinalFeatures, score: float
    ) -> str:
        if score < 15:
            return "Senales oculares cercanas al comportamiento habitual."
        if features.perclos_trend.direction is TrendDirection.INCREASING:
            return "Incremento sostenido de PERCLOS respecto al baseline."
        if (
            features.ear_deviation.absolute_difference is not None
            and features.ear_deviation.absolute_difference < 0
        ):
            return "Disminucion de EAR respecto al comportamiento habitual."
        return "Senales oculares persistentes compatibles con mayor esfuerzo visual."

    def _build_postural_explanation(
        self, features: LongitudinalFeatures, score: float
    ) -> str:
        if score < 15:
            return "Comportamiento postural cercano al baseline personal."
        return "Desviacion postural mantenida respecto al comportamiento habitual."

    def _build_temporal_explanation(self, exposure, score: float) -> str:
        if score < 10:
            return "La exposicion temporal actual aporta poca evidencia por si sola."
        return "Exposicion prolongada combinable con otras senales longitudinales."

    @staticmethod
    def _normalize_positive(value: float | None, reference: float) -> float:
        if value is None or not math.isfinite(value) or reference <= 0:
            return 0.0
        return clamp_score((max(0.0, value) / reference) * 100.0)

    def _positive_difference_score(
        self, deviation: MetricDeviation, *, reference: float
    ) -> float:
        return self._normalize_positive(deviation.absolute_difference, reference)

    def _negative_difference_score(
        self, deviation: MetricDeviation, *, reference: float
    ) -> float:
        if deviation.absolute_difference is None:
            return 0.0
        return self._normalize_positive(-deviation.absolute_difference, reference)

    def _positive_z_score(self, deviation: MetricDeviation, reference: float) -> float:
        return self._normalize_positive(deviation.z_score, reference)

    def _negative_z_score(self, deviation: MetricDeviation, reference: float) -> float:
        if deviation.z_score is None:
            return 0.0
        return self._normalize_positive(-deviation.z_score, reference)

    @staticmethod
    def _persistence_from_count(sample_count: int, target_samples: int) -> float:
        if target_samples <= 0:
            return 0.0
        return clamp_score(min(1.0, max(0.0, sample_count / target_samples)) * 100.0)

    @staticmethod
    def _mean(*values: float) -> float:
        valid_values = [
            value for value in values if value is not None and math.isfinite(value)
        ]
        if not valid_values:
            return 0.0
        return clamp_score(sum(valid_values) / len(valid_values))
