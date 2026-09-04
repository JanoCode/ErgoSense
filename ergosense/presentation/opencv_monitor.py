import time
from datetime import timedelta

import cv2

from ergosense.application.baseline_service import BaselineService
from ergosense.application.session_service import (
    ObservationSampler,
    SessionService,
)
from driver_state_detection.attention_scorer import AttentionScorer
from driver_state_detection.overlays import draw_dashboard
from ergosense.application.live_monitoring import (
    LiveFrameAnalyzer,
    LiveMonitoringService,
)
from ergosense.domain.session import SessionStatus
from ergosense.infrastructure.live_monitoring import LiveFrameSource


def run_opencv_monitoring(
    args,
    *,
    camera_matrix=None,
    dist_coeffs=None,
    camera_image_size=None,
):
    """Run the current OpenCV live monitor using the modular pipeline."""
    source = LiveFrameSource(
        camera=args.camera,
        model_path=args.model_path,
        max_faces=args.max_faces,
        min_detection_confidence=args.min_detection_confidence,
        min_face_presence_confidence=args.min_face_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        show_eye_proc=args.show_eye_proc,
        eye_preview_scale=args.eye_preview_scale,
        show_axis=args.show_axis,
        show_eye_keypoints=args.show_eye_keypoints,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        camera_image_size=camera_image_size,
    )
    scorer = AttentionScorer(
        t_now=time.perf_counter(),
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        perclos_thresh=args.perclos_thresh,
        perclos_time_period=args.perclos_window,
        perclos_min_valid_fraction=args.perclos_min_valid_fraction,
        pose_time_thresh=args.pose_time_thresh,
        decay_factor=args.decay_factor,
        verbose=args.verbose,
    )
    analyzer = LiveFrameAnalyzer(scorer)
    session_service = SessionService()
    session = session_service.start_session()
    baseline_service = BaselineService(
        calibration_duration=timedelta(seconds=args.baseline_calibration_duration)
    )
    baseline_service.start_calibration(session=session)
    sampler = ObservationSampler(
        interval=timedelta(seconds=args.session_sample_interval)
    )
    sampler.start(session.started_at)
    service = LiveMonitoringService(
        source,
        analyzer,
        session_service=session_service,
        observation_sampler=sampler,
        baseline_service=baseline_service,
    )
    stream = service.stream()

    try:
        for frame, result in stream:
            draw_dashboard(
                frame,
                ear=result.observation.ear,
                gaze=result.observation.gaze,
                perclos=result.observation.perclos,
                perclos_ready=result.observation.perclos_ready,
                roll=result.observation.roll,
                pitch=result.observation.pitch,
                yaw=result.observation.yaw,
                ear_threshold=args.ear_thresh,
                gaze_threshold=args.gaze_thresh,
                perclos_threshold=args.perclos_thresh,
                roll_threshold=args.roll_thresh,
                pitch_threshold=args.pitch_thresh,
                yaw_threshold=args.yaw_thresh,
                roll_limit=args.roll_bar_limit,
                pitch_limit=args.pitch_bar_limit,
                yaw_limit=args.yaw_bar_limit,
                alerts=result.alerts,
                face_detected=result.observation.face_detected,
                fps=result.observation.fps if args.show_fps else None,
                processing_ms=(
                    result.observation.processing_ms if args.show_proc_time else None
                ),
                show_panels=args.show_dashboard,
            )

            cv2.imshow("Press 'q' to terminate", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.close()
        baseline_service.finish_calibration()
        if session_service.get_status() is SessionStatus.ACTIVE:
            session_service.end_session()
        cv2.destroyAllWindows()
