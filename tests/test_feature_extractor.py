"""
Tests for Feature Extractor
=============================
Validates feature extraction correctness, invariance properties,
and output dimensionality.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.feature_extractor import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor()


@pytest.fixture
def sample_landmarks():
    """Generate a valid set of 21 landmarks."""
    np.random.seed(42)
    return np.random.rand(21, 3).astype(np.float32)


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""

    def test_output_shape(self, extractor, sample_landmarks):
        """Feature vector should have the expected dimension."""
        features = extractor.extract(sample_landmarks)
        assert features.shape == (config.INPUT_FEATURE_DIM,), \
            f"Expected ({config.INPUT_FEATURE_DIM},), got {features.shape}"

    def test_output_dtype(self, extractor, sample_landmarks):
        """Features should be float32."""
        features = extractor.extract(sample_landmarks)
        assert features.dtype == np.float32

    def test_get_feature_dim(self, extractor):
        """Static feature dim should match config."""
        assert extractor.get_feature_dim() == config.INPUT_FEATURE_DIM

    def test_translation_invariance(self, extractor, sample_landmarks):
        """Features should be invariant to translation."""
        shifted = sample_landmarks + np.array([0.1, 0.2, 0.0])
        f1 = extractor.extract(sample_landmarks)
        f2 = extractor.extract(shifted)
        np.testing.assert_allclose(f1, f2, atol=0.1)

    def test_scale_robustness(self, extractor, sample_landmarks):
        """Features should be robust to uniform scaling."""
        wrist = sample_landmarks[0].copy()
        scaled = (sample_landmarks - wrist) * 1.5 + wrist
        f1 = extractor.extract(sample_landmarks)
        f2 = extractor.extract(scaled)
        # Should be similar (not identical due to non-linear features)
        correlation = np.corrcoef(f1, f2)[0, 1]
        assert correlation > 0.8, f"Correlation {correlation} too low after scaling"

    def test_left_right_mirroring(self, extractor, sample_landmarks):
        """Left and right hand features should be comparable after mirroring."""
        f_right = extractor.extract(sample_landmarks, handedness="Right")
        f_left = extractor.extract(sample_landmarks, handedness="Left")
        # They won't be identical but should have the same shape
        assert f_right.shape == f_left.shape

    def test_no_nan_or_inf(self, extractor, sample_landmarks):
        """Features should not contain NaN or Inf values."""
        features = extractor.extract(sample_landmarks)
        assert not np.any(np.isnan(features)), "Features contain NaN"
        assert not np.any(np.isinf(features)), "Features contain Inf"

    def test_zero_landmarks_handled(self, extractor):
        """Should handle degenerate (all-zero) landmarks gracefully."""
        zeros = np.zeros((21, 3), dtype=np.float32)
        features = extractor.extract(zeros)
        assert features.shape == (config.INPUT_FEATURE_DIM,)
        assert not np.any(np.isnan(features))
