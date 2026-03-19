"""
Utility Functions
=================
Shared helper functions used across the gesture recognition system.
"""

import time
import numpy as np
from collections import deque
from typing import Tuple

import cv2


class FPSCounter:
    """
    Real-time FPS counter using a sliding window of timestamps.

    Uses a deque of the last N frame timestamps to compute a stable,
    moving-average FPS rather than a noisy instantaneous value.
    """

    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)

    def tick(self):
        """Record a new frame timestamp."""
        self.timestamps.append(time.perf_counter())

    def get_fps(self) -> float:
        """Return current FPS estimate."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


class LatencyTracker:
    """
    Tracks end-to-end latency of each pipeline invocation.

    Provides running statistics (mean, p50, p95, p99) over a window.
    """

    def __init__(self, window_size: int = 100):
        self.latencies = deque(maxlen=window_size)
        self._start = None

    def start(self):
        """Mark the start of a pipeline invocation."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """Mark the end and record the latency. Returns latency in ms."""
        if self._start is None:
            return 0.0
        latency_ms = (time.perf_counter() - self._start) * 1000
        self.latencies.append(latency_ms)
        self._start = None
        return latency_ms

    def get_stats(self) -> dict:
        """Return latency statistics in milliseconds."""
        if not self.latencies:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        arr = np.array(self.latencies)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 5,
) -> np.ndarray:
    """
    Draw text with a filled background rectangle for readability.

    Parameters
    ----------
    frame : np.ndarray
        Image to draw on (modified in-place).
    text : str
        Text string to render.
    position : Tuple[int, int]
        Bottom-left corner of the text.
    font_scale : float
        Font scale factor.
    color : Tuple[int, int, int]
        Text color (BGR).
    bg_color : Tuple[int, int, int]
        Background rectangle color (BGR).
    thickness : int
        Text thickness.
    padding : int
        Padding around text inside the rectangle.

    Returns
    -------
    np.ndarray
        The frame with text drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_confidence_bar(
    frame: np.ndarray,
    label: str,
    confidence: float,
    position: Tuple[int, int],
    bar_width: int = 150,
    bar_height: int = 18,
) -> np.ndarray:
    """
    Draw a labeled confidence bar on the frame.

    Parameters
    ----------
    frame : np.ndarray
        Image to draw on (modified in-place).
    label : str
        Gesture label text.
    confidence : float
        Confidence value between 0 and 1.
    position : Tuple[int, int]
        Top-left corner of the bar.
    bar_width : int
        Total width of the bar in pixels.
    bar_height : int
        Height of the bar in pixels.

    Returns
    -------
    np.ndarray
        The frame with the confidence bar drawn.
    """
    x, y = position
    conf_pct = max(0.0, min(1.0, confidence))

    # Background bar
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), cv2.FILLED)

    # Filled portion — color shifts from red to green based on confidence
    fill_width = int(bar_width * conf_pct)
    if conf_pct > 0.7:
        fill_color = (0, 200, 0)     # Green
    elif conf_pct > 0.4:
        fill_color = (0, 200, 200)   # Yellow
    else:
        fill_color = (0, 0, 200)     # Red
    cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), fill_color, cv2.FILLED)

    # Border
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), 1)

    # Label text
    label_text = f"{label}: {conf_pct:.0%}"
    cv2.putText(
        frame, label_text, (x + 5, y + bar_height - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return frame
