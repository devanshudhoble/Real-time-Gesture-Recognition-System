"""
Gesture Classifier Model
=========================
A lightweight Multi-Layer Perceptron (MLP) for gesture classification.

Architecture Choice — Why MLP over CNN / Vision Transformer:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Our input is a 78-dim STRUCTURED FEATURE VECTOR, not raw       │
    │ pixel data. CNNs excel at spatial hierarchies in images;       │
    │ Vision Transformers shine with patch-based attention on images. │
    │ Neither architecture advantage applies to 1-D landmark-derived │
    │ features. A 3-layer MLP achieves >95% accuracy on this task    │
    │ with <1ms inference and <1MB model size — far better suited    │
    │ for real-time edge deployment than heavyweight alternatives.    │
    └─────────────────────────────────────────────────────────────────┘

Model Details:
    Input  → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
           → Linear(num_classes)
    Output → raw logits (softmax applied externally for flexibility)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config


class GestureClassifier(nn.Module):
    """
    Lightweight MLP for hand gesture classification.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector. Default: config.INPUT_FEATURE_DIM.
    hidden_dim_1 : int
        Units in first hidden layer. Default: config.HIDDEN_DIM_1.
    hidden_dim_2 : int
        Units in second hidden layer. Default: config.HIDDEN_DIM_2.
    num_classes : int
        Number of gesture classes. Default: config.NUM_CLASSES.
    dropout_rate : float
        Dropout probability. Default: config.DROPOUT_RATE.
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_FEATURE_DIM,
        hidden_dim_1: int = config.HIDDEN_DIM_1,
        hidden_dim_2: int = config.HIDDEN_DIM_2,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Layer 2
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Output layer
            nn.Linear(hidden_dim_2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (batch_size, num_classes).
            Apply softmax externally for probabilities.
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> tuple:
        """
        Predict gesture class and confidence.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (1, input_dim) or (batch_size, input_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (predicted_class_indices, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)
        return predicted, confidence

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return full probability distribution over classes.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Probabilities of shape (batch_size, num_classes).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Return approximate model size in megabytes."""
        param_size = sum(
            p.nelement() * p.element_size() for p in self.parameters()
        )
        buffer_size = sum(
            b.nelement() * b.element_size() for b in self.buffers()
        )
        return (param_size + buffer_size) / (1024 ** 2)

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "GestureClassifier":
        """
        Load a trained model from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the .pth checkpoint file.
        device : str
            Device to load onto ("cpu" or "cuda").

        Returns
        -------
        GestureClassifier
            Loaded model in eval mode.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_dim=checkpoint.get("input_dim", config.INPUT_FEATURE_DIM),
            num_classes=checkpoint.get("num_classes", config.NUM_CLASSES),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
