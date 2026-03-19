"""
Tests for Post-Processor and Pipeline Integration
===================================================
Tests temporal smoothing, confidence thresholding, hysteresis,
and data generation pipeline.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.post_processor import PostProcessor
from gestures.feature_extractor import FeatureExtractor


@pytest.fixture
def post_processor():
    return PostProcessor(
        ema_alpha=0.6,
        confidence_threshold=0.5,
        hysteresis_frames=2,
    )


class TestPostProcessor:
    """Tests for the PostProcessor class."""

    def test_first_call_returns_result(self, post_processor):
        """First call should return a result."""
        probs = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        probs[4] = 0.9  # open_palm
        gesture, conf, smoothed = post_processor.process(probs, hand_id="Right")
        assert gesture is not None
        assert conf > 0.5

    def test_low_confidence_returns_none(self, post_processor):
        """Low confidence predictions should return None gesture."""
        probs = np.ones(config.NUM_CLASSES, dtype=np.float32) / config.NUM_CLASSES
        gesture, conf, smoothed = post_processor.process(probs, hand_id="Right")
        assert gesture is None or conf < 0.5

    def test_ema_smoothing(self, post_processor):
        """EMA should smooth rapid changes."""
        probs1 = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        probs1[0] = 1.0  # point_left

        probs2 = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        probs2[1] = 1.0  # point_right

        # First call
        post_processor.process(probs1, hand_id="Test")

        # Second call — smoothed probs should still have some weight on class 0
        _, _, smoothed = post_processor.process(probs2, hand_id="Test")
        assert smoothed[0] > 0, "EMA should retain some probability from previous frame"

    def test_reset(self, post_processor):
        """Reset should clear all state."""
        probs = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        probs[4] = 0.9
        post_processor.process(probs, hand_id="Right")
        post_processor.reset()
        # After reset, state should be empty
        assert len(post_processor._hand_states) == 0

    def test_multiple_hands(self, post_processor):
        """Should maintain separate state for each hand."""
        probs_left = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        probs_left[4] = 0.9  # open_palm

        probs_right = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        probs_right[5] = 0.9  # thumbs_up

        g_left, _, _ = post_processor.process(probs_left, hand_id="Left")
        g_right, _, _ = post_processor.process(probs_right, hand_id="Right")

        # Different gestures for different hands
        assert g_left != g_right or (g_left is None and g_right is None)


class TestDataPipeline:
    """Tests for the data generation pipeline."""

    def test_feature_dim_consistency(self):
        """Feature extractor dim should match config."""
        extractor = FeatureExtractor()
        assert extractor.get_feature_dim() == config.INPUT_FEATURE_DIM

    def test_generate_sample(self):
        """Should be able to generate and extract features from a random landmark set."""
        extractor = FeatureExtractor()
        landmarks = np.random.rand(21, 3).astype(np.float32)
        features = extractor.extract(landmarks)
        assert features.shape == (config.INPUT_FEATURE_DIM,)
        assert features.dtype == np.float32


class TestConfig:
    """Tests for configuration consistency."""

    def test_gesture_classes_count(self):
        """NUM_CLASSES should match GESTURE_CLASSES length."""
        assert config.NUM_CLASSES == len(config.GESTURE_CLASSES)

    def test_gesture_mappings(self):
        """Forward and reverse mappings should be consistent."""
        for name, idx in config.GESTURE_TO_IDX.items():
            assert config.IDX_TO_GESTURE[idx] == name

    def test_splits_sum_to_one(self):
        """Train/val/test splits should sum to 1."""
        total = config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT
        assert abs(total - 1.0) < 1e-6
