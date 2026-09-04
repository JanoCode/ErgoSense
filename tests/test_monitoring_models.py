from datetime import datetime, timezone
from uuid import uuid4

from ergosense.domain.monitoring import AnalysisResult, AnalysisState, FrameObservation


def test_frame_observation_keeps_runtime_metrics_together():
    observation = FrameObservation(
        timestamp=12.5,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        face_detected=True,
        ear=0.21,
        gaze=0.11,
        roll=-2.0,
        pitch=1.0,
        yaw=3.0,
        fps=29.9,
        processing_ms=14.2,
    )

    assert observation.timestamp == 12.5
    assert observation.observed_at.tzinfo is timezone.utc
    assert observation.face_detected
    assert observation.ear == 0.21
    assert observation.processing_ms == 14.2


def test_analysis_result_groups_state_alerts_and_perclos():
    observation = FrameObservation(
        timestamp=1.0,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        face_detected=True,
    )
    state = AnalysisState(tired=True, distracted=True)
    session_id = uuid4()

    result = AnalysisResult(
        observation=observation,
        state=state,
        session_id=session_id,
        alerts=("TIRED", "DISTRACTED"),
        perclos=0.3,
        perclos_ready=True,
    )

    assert result.observation is observation
    assert result.state.tired
    assert result.state.distracted
    assert result.session_id == session_id
    assert result.alerts == ("TIRED", "DISTRACTED")
    assert result.perclos == 0.3
    assert result.perclos_ready
