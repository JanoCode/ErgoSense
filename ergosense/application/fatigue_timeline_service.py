from datetime import timedelta
from typing import Iterable

from ergosense.domain.assessment import FatigueAssessment
from ergosense.domain.fatigue_timeline import FatigueTimeline


class FatigueTimelineService:
    """Build and inspect in-memory timelines of successive assessments."""

    def create_timeline(self, session_id) -> FatigueTimeline:
        return FatigueTimeline(session_id=session_id)

    def add_assessment(
        self, timeline: FatigueTimeline, assessment: FatigueAssessment
    ) -> FatigueTimeline:
        return timeline.add(assessment)

    def build_timeline(
        self, assessments: Iterable[FatigueAssessment]
    ) -> FatigueTimeline:
        iterator = iter(assessments)
        try:
            first = next(iterator)
        except StopIteration:
            raise ValueError("at least one assessment is required to build a timeline")
        timeline = self.create_timeline(first.session_id)
        timeline = self.add_assessment(timeline, first)
        for assessment in iterator:
            timeline = self.add_assessment(timeline, assessment)
        return timeline

    @staticmethod
    def get_latest(timeline: FatigueTimeline):
        return timeline.latest

    @staticmethod
    def get_duration(timeline: FatigueTimeline) -> timedelta:
        return timeline.duration
