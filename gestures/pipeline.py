"""
Gesture Recognition Pipeline
==============================
End-to-end orchestrator that chains together all components:

    VideoFrame → HandDetector → FeatureExtractor → Classifier → PostProcessor → Result

Handles:
    - Multiple hands in a single frame
    - Edge cases (no hands, missing landmarks, partial occlusion)
    - Both PyTorch and ONNX Runtime inference backends
    - FPS and latency tracking
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config

from gestures.hand_detector import HandDetector, HandResult
from gestures.feature_extractor import FeatureExtractor
from gestures.classifier import GestureClassifier
from gestures.post_processor import PostProcessor
from gestures.utils import FPSCounter, LatencyTracker


@dataclass
class GestureResult:
    """Result for a single hand in a single frame."""
    gesture: Optional[str]              # Predicted gesture name (None if low confidence)
    confidence: float                    # Confidence score [0, 1]
    all_probabilities: np.ndarray        # Full smoothed probability distribution
    handedness: str                      # "Left" or "Right"
    handedness_score: float              # MediaPipe handedness confidence
    landmarks: np.ndarray                # (21, 3) normalized landmarks
    bbox: Tuple[int, int, int, int]     # Bounding box (x1, y1, x2, y2)


class GestureRecognitionPipeline:
    """
    Complete gesture recognition pipeline.

    Orchestrates hand detection, feature extraction, classification,
    and post-processing into a single `process_frame` call.

    Parameters
    ----------
    model_path : str, optional
        Path to saved PyTorch model (.pth). If None, uses default path.
    use_onnx : bool
        If True, use ONNX Runtime for inference (faster on CPU).
    device : str
        PyTorch device ("cpu" or "cuda").
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_onnx: bool = False,
        device: str = "cpu",
    ):
        self.device = device
        self.use_onnx = use_onnx

        # Initialize components
        self.detector = HandDetector()
        self.feature_extractor = FeatureExtractor()
        self.post_processor = PostProcessor()

        # Load model
        if use_onnx:
            self._load_onnx_model(model_path or str(config.ONNX_MODEL_PATH))
        else:
            self._load_pytorch_model(model_path or str(config.BEST_MODEL_PATH))

        # Performance tracking
        self.fps_counter = FPSCounter()
        self.latency_tracker = LatencyTracker()

    def _load_pytorch_model(self, path: str):
        """Load PyTorch model from checkpoint."""
        self.model = GestureClassifier.load_from_checkpoint(path, self.device)
        self.model.eval()

    def _load_onnx_model(self, path: str):
        """Load ONNX model for inference."""
        import onnxruntime as ort
        self.ort_session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
        )
        self.ort_input_name = self.ort_session.get_inputs()[0].name

    def process_frame(self, frame: np.ndarray) -> List[GestureResult]:
        """
        Process a single video frame through the full pipeline.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3) from webcam / video.

        Returns
        -------
        List[GestureResult]
            One result per detected hand.
        """
        self.latency_tracker.start()
        self.fps_counter.tick()

        # Step 1: Detect hands
        hands: List[HandResult] = self.detector.detect(frame)

        results: List[GestureResult] = []

        for hand in hands:
            try:
                # Step 2: Extract features
                features = self.feature_extractor.extract(
                    hand.landmarks, hand.handedness
                )

                # Step 3: Classify
                raw_probs = self._classify(features)

                # Step 4: Post-process (smooth, threshold, hysteresis)
                gesture, confidence, smoothed_probs = self.post_processor.process(
                    raw_probs=raw_probs,
                    landmarks=hand.landmarks,
                    hand_id=hand.handedness,
                )

                results.append(GestureResult(
                    gesture=gesture,
                    confidence=confidence,
                    all_probabilities=smoothed_probs,
                    handedness=hand.handedness,
                    handedness_score=hand.handedness_score,
                    landmarks=hand.landmarks,
                    bbox=hand.bbox,
                ))
            except Exception:
                # Gracefully handle partial occlusion or bad landmarks
                continue

        self.latency_tracker.stop()
        return results

    def _classify(self, features: np.ndarray) -> np.ndarray:
        """
        Run classification on a feature vector.

        Returns
        -------
        np.ndarray
            Softmax probability distribution over gesture classes.
        """
        if self.use_onnx:
            return self._classify_onnx(features)
        else:
            return self._classify_pytorch(features)

    def _classify_pytorch(self, features: np.ndarray) -> np.ndarray:
        """PyTorch inference."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.model.predict_proba(x)
        return probs.cpu().numpy().squeeze()

    def _classify_onnx(self, features: np.ndarray) -> np.ndarray:
        """ONNX Runtime inference."""
        x = features.reshape(1, -1).astype(np.float32)
        outputs = self.ort_session.run(None, {self.ort_input_name: x})
        logits = outputs[0].squeeze()
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def get_fps(self) -> float:
        """Return current FPS."""
        return self.fps_counter.get_fps()

    def get_latency_stats(self) -> dict:
        """Return latency statistics in ms."""
        return self.latency_tracker.get_stats()

    def reset(self):
        """Reset post-processor state (e.g., after hand leaves frame)."""
        self.post_processor.reset()

    def release(self):
        """Release all resources."""
        self.detector.release()
