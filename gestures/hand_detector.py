"""
Hand Detector Module
====================
Wraps MediaPipe Hand Landmarker (Tasks API) to detect and track hands in
video frames. Supports multiple hands, configurable confidence thresholds,
and returns structured landmark data for downstream feature extraction.

Design Rationale:
    MediaPipe Hands provides a state-of-the-art, lightweight hand detection
    model that runs efficiently on CPU. It outputs 21 3D landmarks per hand,
    which is ideal for landmark-based gesture classification. This approach
    is preferred over training a custom detector because it works reliably
    across lighting conditions and skin tones out of the box.

Note:
    MediaPipe >= 0.10.x uses the new Tasks API (mp.tasks.vision.HandLandmarker)
    instead of the legacy mp.solutions.hands API.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


@dataclass
class HandResult:
    """Structured result for a single detected hand."""
    landmarks: np.ndarray            # (21, 3) — normalized x, y, z
    world_landmarks: np.ndarray      # (21, 3) — real-world x, y, z in meters
    handedness: str                   # "Left" or "Right"
    handedness_score: float           # Confidence of handedness classification
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in pixels


# MediaPipe hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle  (via 0→9 approximation)
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


class HandDetector:
    """
    MediaPipe-based hand detector using the Tasks API.

    Handles:
        - Multiple hands (up to max_num_hands)
        - Varying lighting via configurable confidence thresholds
        - RGB/BGR conversion
        - Bounding-box computation from landmarks

    Parameters
    ----------
    max_num_hands : int
        Maximum number of hands to detect. Default from config.
    min_detection_confidence : float
        Minimum confidence for initial detection. Default from config.
    min_tracking_confidence : float
        Minimum confidence for frame-to-frame tracking. Default from config.
    model_path : str, optional
        Path to the hand_landmarker.task file. Auto-detected if None.
    """

    def __init__(
        self,
        max_num_hands: int = config.MEDIAPIPE_MAX_HANDS,
        min_detection_confidence: float = config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        model_path: str = None,
    ):
        if model_path is None:
            model_path = str(config.MODELS_DIR / "hand_landmarker.task")

        # Ensure model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}\n"
                f"Download it with:\n"
                f"  Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' "
                f"-OutFile '{model_path}'"
            )

        # Create HandLandmarker with VIDEO running mode
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0


    # ─────────────────────── public API ───────────────────────

    def detect(self, frame: np.ndarray) -> List[HandResult]:
        """
        Detect hands in a BGR video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3) as returned by OpenCV.

        Returns
        -------
        List[HandResult]
            One result per detected hand, empty list if none found.
        """
        h, w, _ = frame.shape

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect using VIDEO mode (tracks hands, significantly faster)
        # We need strictly increasing timestamps
        import time
        current_time_ms = int(time.time() * 1000)
        if current_time_ms <= self._frame_timestamp_ms:
            current_time_ms = self._frame_timestamp_ms + 1
        self._frame_timestamp_ms = current_time_ms

        result = self.landmarker.detect_for_video(mp_image, current_time_ms)


        hands: List[HandResult] = []

        if not result.hand_landmarks:
            return hands

        for i in range(len(result.hand_landmarks)):
            hand_lms = result.hand_landmarks[i]
            hand_world_lms = result.hand_world_landmarks[i]
            handedness_list = result.handedness[i]

            # Normalized landmarks → numpy (21, 3)
            lm_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms],
                dtype=np.float32,
            )

            # World landmarks → numpy (21, 3) in metres
            wlm_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_world_lms],
                dtype=np.float32,
            )

            # Handedness
            handedness = handedness_list[0].category_name  # "Left" or "Right"
            handedness_score = handedness_list[0].score

            # Bounding box from normalised landmarks
            x_coords = (lm_array[:, 0] * w).astype(int)
            y_coords = (lm_array[:, 1] * h).astype(int)
            margin = 20
            bbox = (
                max(0, int(x_coords.min()) - margin),
                max(0, int(y_coords.min()) - margin),
                min(w, int(x_coords.max()) + margin),
                min(h, int(y_coords.max()) + margin),
            )

            hands.append(HandResult(
                landmarks=lm_array,
                world_landmarks=wlm_array,
                handedness=handedness,
                handedness_score=handedness_score,
                bbox=bbox,
            ))

        return hands

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_result: HandResult,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """
        Draw hand landmarks and optional bounding box on the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to draw on (modified in-place and returned).
        hand_result : HandResult
            Detection result for one hand.
        draw_bbox : bool
            Whether to draw the bounding-box rectangle.

        Returns
        -------
        np.ndarray
            The frame with annotations.
        """
        return self.draw_landmarks_data(
            frame=frame,
            landmarks=hand_result.landmarks,
            bbox=hand_result.bbox,
            draw_bbox=draw_bbox,
        )

    def draw_landmarks_data(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        bbox: Tuple[int, int, int, int] = None,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """
        Draw landmarks and an optional bounding box from raw landmark data.

        This variant is useful for downstream pipeline results that do not
        carry the full HandResult object.
        """
        h, w, _ = frame.shape

        # Draw individual landmarks
        for i, (x, y, z) in enumerate(landmarks):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Draw connections
        for conn in HAND_CONNECTIONS:
            start = landmarks[conn[0]]
            end = landmarks[conn[1]]
            pt1 = (int(start[0] * w), int(start[1] * h))
            pt2 = (int(end[0] * w), int(end[1] * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Bounding box
        if draw_bbox and bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame

    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __del__(self):
        self.release()
