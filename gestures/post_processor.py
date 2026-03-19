"""
Post-Processor Module
=====================
Applies temporal smoothing and confidence filtering to raw classifier outputs
to produce stable, reliable gesture predictions in real-time.

Techniques:
    1. Exponential Moving Average (EMA) on softmax probabilities
       → Smooths out noisy frame-to-frame predictions
    2. Confidence thresholding
       → Only reports gesture if confidence exceeds threshold
    3. Hysteresis (consecutive-frame confirmation)
       → Requires N consecutive frames predicting same gesture before switching
    4. Wave detection via temporal pattern analysis
       → Detects oscillating open-palm positions across a sliding window
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config


class PostProcessor:
    """
    Temporal post-processor for gesture classification.

    Provides smooth, stable gesture predictions from noisy per-frame outputs
    by combining EMA smoothing, confidence gating, and hysteresis.

    Parameters
    ----------
    ema_alpha : float
        Smoothing factor (0 < α ≤ 1). Higher = more responsive, lower = smoother.
    confidence_threshold : float
        Minimum smoothed confidence to accept a gesture.
    hysteresis_frames : int
        Number of consecutive agreeing frames before switching gesture.
    wave_window_size : int
        Number of frames in the wave detection sliding window.
    wave_min_oscillations : int
        Minimum direction reversals to classify as "wave".
    """

    def __init__(
        self,
        ema_alpha: float = config.EMA_ALPHA,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        hysteresis_frames: int = config.HYSTERESIS_FRAMES,
        wave_window_size: int = config.WAVE_WINDOW_SIZE,
        wave_min_oscillations: int = config.WAVE_MIN_OSCILLATIONS,
    ):
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold
        self.hysteresis_frames = hysteresis_frames
        self.wave_window_size = wave_window_size
        self.wave_min_oscillations = wave_min_oscillations

        # State per hand (keyed by handedness string)
        self._hand_states = {}

    def _get_state(self, hand_id: str) -> dict:
        """Get or create state for a specific hand."""
        if hand_id not in self._hand_states:
            self._hand_states[hand_id] = {
                "ema_probs": None,           # Smoothed probability vector
                "current_gesture": None,     # Currently displayed gesture
                "candidate_gesture": None,   # Candidate awaiting hysteresis
                "candidate_count": 0,        # Consecutive frames for candidate
                "landmark_history": deque(maxlen=self.wave_window_size),
            }
        return self._hand_states[hand_id]

    def process(
        self,
        raw_probs: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        hand_id: str = "Right",
    ) -> Tuple[Optional[str], float, np.ndarray]:
        """
        Process raw classifier probabilities through the smoothing pipeline.

        Parameters
        ----------
        raw_probs : np.ndarray
            Raw softmax probabilities from classifier, shape (num_classes,).
        landmarks : np.ndarray, optional
            (21, 3) landmarks for wave detection.
        hand_id : str
            Identifier for the hand ("Left" or "Right").

        Returns
        -------
        Tuple[Optional[str], float, np.ndarray]
            (gesture_name, confidence, smoothed_probs)
            gesture_name is None if confidence below threshold.
        """
        state = self._get_state(hand_id)

        # ── 1. EMA smoothing ──
        if state["ema_probs"] is None:
            state["ema_probs"] = raw_probs.copy()
        else:
            state["ema_probs"] = (
                self.ema_alpha * raw_probs
                + (1 - self.ema_alpha) * state["ema_probs"]
            )

        smoothed = state["ema_probs"]
        pred_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[pred_idx])

        # ── 2. Wave detection (temporal pattern) ──
        if landmarks is not None:
            state["landmark_history"].append(landmarks.copy())
            if self._detect_wave(state):
                pred_idx = config.GESTURE_TO_IDX["wave"]
                confidence = max(confidence, 0.75)  # Boost wave confidence

        # ── 3. Confidence threshold ──
        if confidence < self.confidence_threshold:
            return None, confidence, smoothed

        # ── 4. Hysteresis ──
        gesture_name = config.IDX_TO_GESTURE[pred_idx]

        if gesture_name == state["current_gesture"]:
            # Same gesture — reset candidate
            state["candidate_gesture"] = None
            state["candidate_count"] = 0
        elif gesture_name == state["candidate_gesture"]:
            # Same candidate — increment count
            state["candidate_count"] += 1
            if state["candidate_count"] >= self.hysteresis_frames:
                state["current_gesture"] = gesture_name
                state["candidate_gesture"] = None
                state["candidate_count"] = 0
            else:
                # Still waiting for hysteresis — return old gesture
                if state["current_gesture"] is not None:
                    gesture_name = state["current_gesture"]
                    confidence = float(smoothed[config.GESTURE_TO_IDX[gesture_name]])
        else:
            # New candidate
            state["candidate_gesture"] = gesture_name
            state["candidate_count"] = 1
            if state["current_gesture"] is not None:
                gesture_name = state["current_gesture"]
                confidence = float(smoothed[config.GESTURE_TO_IDX[gesture_name]])
            else:
                state["current_gesture"] = gesture_name

        return gesture_name, confidence, smoothed

    def _detect_wave(self, state: dict) -> bool:
        """
        Detect wave gesture by analyzing lateral oscillation of the wrist
        landmark over a temporal window.

        Wave = hand moving left-right (or right-left) repeatedly.
        We track the x-coordinate of the wrist and count direction reversals.
        """
        history = state["landmark_history"]
        if len(history) < self.wave_window_size:
            return False

        # Extract wrist x-coordinates over time
        wrist_xs = [lm[config.WRIST][0] for lm in history]

        # Compute direction changes
        directions = []
        for i in range(1, len(wrist_xs)):
            diff = wrist_xs[i] - wrist_xs[i - 1]
            if abs(diff) > 0.005:  # Minimum movement threshold
                directions.append(1 if diff > 0 else -1)

        if len(directions) < 3:
            return False

        # Count direction reversals
        reversals = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i - 1]:
                reversals += 1

        # Also check that current gesture is open_palm-like
        # (wave is an open palm moving side to side)
        current_lm = history[-1]
        # Simple heuristic: all fingertips should be above their MCPs (extended)
        tips_extended = 0
        tip_indices = [config.INDEX_TIP, config.MIDDLE_TIP, config.RING_TIP, config.PINKY_TIP]
        mcp_indices = [config.INDEX_MCP, config.MIDDLE_MCP, config.RING_MCP, config.PINKY_MCP]
        for tip_idx, mcp_idx in zip(tip_indices, mcp_indices):
            if current_lm[tip_idx][1] < current_lm[mcp_idx][1]:  # y decreases upward
                tips_extended += 1

        return reversals >= self.wave_min_oscillations and tips_extended >= 3

    def reset(self, hand_id: Optional[str] = None):
        """
        Reset smoothing state.

        Parameters
        ----------
        hand_id : str, optional
            Reset specific hand. If None, reset all.
        """
        if hand_id is None:
            self._hand_states.clear()
        elif hand_id in self._hand_states:
            del self._hand_states[hand_id]
