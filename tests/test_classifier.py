"""
Tests for Gesture Classifier Model
====================================
Validates model architecture, forward pass, prediction methods,
and checkpoint loading.
"""

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.classifier import GestureClassifier


@pytest.fixture
def model():
    return GestureClassifier()


@pytest.fixture
def sample_input():
    """Single sample input tensor."""
    return torch.randn(1, config.INPUT_FEATURE_DIM)


@pytest.fixture
def batch_input():
    """Batch of input tensors."""
    return torch.randn(16, config.INPUT_FEATURE_DIM)


class TestGestureClassifier:
    """Tests for the GestureClassifier model."""

    def test_forward_output_shape(self, model, sample_input):
        """Forward pass should return logits with correct shape."""
        model.eval()  # BatchNorm1d requires batch>1 in train mode
        output = model(sample_input)
        assert output.shape == (1, config.NUM_CLASSES), \
            f"Expected (1, {config.NUM_CLASSES}), got {output.shape}"

    def test_batch_forward(self, model, batch_input):
        """Forward pass should handle batches."""
        output = model(batch_input)
        assert output.shape == (16, config.NUM_CLASSES)

    def test_predict_returns_class_and_confidence(self, model, sample_input):
        """predict() should return (class_index, confidence)."""
        predicted, confidence = model.predict(sample_input)
        assert predicted.shape == (1,)
        assert confidence.shape == (1,)
        assert 0 <= confidence.item() <= 1.0
        assert 0 <= predicted.item() < config.NUM_CLASSES

    def test_predict_proba_sums_to_one(self, model, sample_input):
        """Predicted probabilities should sum to 1."""
        probs = model.predict_proba(sample_input)
        assert probs.shape == (1, config.NUM_CLASSES)
        total = probs.sum(dim=-1).item()
        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, not 1.0"

    def test_predict_proba_non_negative(self, model, sample_input):
        """All probabilities should be non-negative."""
        probs = model.predict_proba(sample_input)
        assert (probs >= 0).all()

    def test_model_size_small(self, model):
        """Model should be well under 500MB requirement."""
        size_mb = model.model_size_mb()
        assert size_mb < 500, f"Model size {size_mb:.2f} MB exceeds 500 MB limit"
        assert size_mb < 1, f"Model should be < 1 MB, got {size_mb:.2f} MB"

    def test_parameter_count(self, model):
        """Model should have a reasonable number of parameters."""
        params = model.count_parameters()
        assert params > 0
        assert params < 1_000_000, f"Too many parameters: {params}"

    def test_no_nan_in_output(self, model, sample_input):
        """Output should not contain NaN values."""
        model.eval()  # BatchNorm1d requires batch>1 in train mode
        output = model(sample_input)
        assert not torch.any(torch.isnan(output))

    def test_eval_mode_deterministic(self, model, sample_input):
        """In eval mode, repeated forward passes should give same output."""
        model.eval()
        with torch.no_grad():
            out1 = model(sample_input)
            out2 = model(sample_input)
        torch.testing.assert_close(out1, out2)

    def test_custom_dimensions(self):
        """Model should work with custom input/output dimensions."""
        custom_model = GestureClassifier(input_dim=50, num_classes=5)
        x = torch.randn(4, 50)
        out = custom_model(x)
        assert out.shape == (4, 5)
