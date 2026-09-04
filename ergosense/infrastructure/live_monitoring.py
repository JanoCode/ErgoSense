import time
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import mediapipe as mp

from driver_state_detection.eye_detector import EyeDetector
from driver_state_detection.pose_estimation import HeadPoseEstimator
from driver_state_detection.utils import get_landmarks
from ergosense.domain.monitoring import FrameObservation


@dataclass
class LiveFrameSample:
    """Raw frame plus the observation extracted from it."""

    frame: object
    observation: FrameObservation
    processing_started_at: float


class LiveFrameSource:
    """Capture frames and extract vision observations for live monitoring."""

    def __init__(
        self,
        *,
        camera: int,
        model_path,
        max_faces: int,
        min_detection_confidence: float,
        min_face_presence_confidence: float,
        min_tracking_confidence: float,
        show_eye_proc: bool,
        eye_preview_scale: float,
        show_axis: bool,
        show_eye_keypoints: bool,
        camera_matrix=None,
        dist_coeffs=None,
        camera_image_size=None,
    ):
        self.camera = camera
        self.show_eye_keypoints = show_eye_keypoints
        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.eye_detector = EyeDetector(
            show_processing=show_eye_proc,
            preview_scale=eye_preview_scale,
        )
        self.head_pose = HeadPoseEstimator(
            show_axis=show_axis,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            camera_image_size=camera_image_size,
        )

    def stream(self):
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            cap.release()
            raise SystemExit(f"Cannot open camera {self.camera}")

        previous_timestamp_ms = -1
        now = time.perf_counter()
        previous_frame_time = now
        stream_start = now
        fps = 0.0

        try:
            with mp.tasks.vision.FaceLandmarker.create_from_options(
                self.options
            ) as detector:
                while True:
                    frame_time = time.perf_counter()
                    elapsed = frame_time - previous_frame_time
                    previous_frame_time = frame_time
                    if elapsed > 0:
                        fps = 1.0 / elapsed

                    received, frame = cap.read()
                    if not received:
                        print("Cannot receive frame from camera or stream ended")
                        break
                    if self.camera == 0:
                        frame = cv2.flip(frame, 1)

                    processing_started_at = time.perf_counter()
                    frame_size = (frame.shape[1], frame.shape[0])
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = max(
                        previous_timestamp_ms + 1,
                        int((frame_time - stream_start) * 1000),
                    )
                    previous_timestamp_ms = timestamp_ms
                    result = detector.detect_for_video(mp_image, timestamp_ms)

                    observation = self._build_observation(
                        frame=frame,
                        frame_time=frame_time,
                        frame_size=frame_size,
                        fps=fps,
                        face_landmarks=result.face_landmarks,
                    )
                    yield LiveFrameSample(
                        frame=frame,
                        observation=observation,
                        processing_started_at=processing_started_at,
                    )
        finally:
            cap.release()

    def _build_observation(self, *, frame, frame_time, frame_size, fps, face_landmarks):
        if not face_landmarks:
            return FrameObservation(
                timestamp=frame_time,
                observed_at=datetime.now(timezone.utc),
                face_detected=False,
                fps=fps,
            )

        landmarks = get_landmarks(face_landmarks)
        if self.show_eye_keypoints:
            self.eye_detector.show_eye_keypoints(frame, landmarks, frame_size)
        ear = self.eye_detector.get_EAR(landmarks, frame_size)
        gaze = self.eye_detector.get_Gaze_Score(frame, landmarks, frame_size)
        posed_frame, roll, pitch, yaw = self.head_pose.get_pose(
            frame, landmarks, frame_size
        )
        if posed_frame is not None:
            frame[:] = posed_frame

        return FrameObservation(
            timestamp=frame_time,
            observed_at=datetime.now(timezone.utc),
            face_detected=True,
            ear=ear,
            gaze=gaze,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            fps=fps,
        )
