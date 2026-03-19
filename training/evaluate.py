"""
Model Evaluation Script
========================
Evaluates the trained gesture classifier on the held-out test set.

Outputs:
    - Classification report (precision, recall, F1-score per class)
    - Confusion matrix visualization
    - Overall accuracy and per-class metrics
    - Saves results to models/ directory

Usage:
    python training/evaluate.py
"""

import numpy as np
import torch
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.classifier import GestureClassifier


def _require_file(path: Path, reason: str):
    """Fail early with a clear message for missing artifacts."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            f"Run {reason} first."
        )


def evaluate():
    """Run evaluation on the test set."""
    print("=" * 60)
    print("Gesture Classifier Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_features_path = config.MODELS_DIR / "test_features.npy"
    test_labels_path = config.MODELS_DIR / "test_labels.npy"
    _require_file(config.BEST_MODEL_PATH, "training/train.py")
    _require_file(test_features_path, "training/train.py")
    _require_file(test_labels_path, "training/train.py")

    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)
    print(f"Test samples: {len(test_features)}")

    # Load model
    model = GestureClassifier.load_from_checkpoint(str(config.BEST_MODEL_PATH), device)
    model.eval()
    print(f"Model loaded from: {config.BEST_MODEL_PATH}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.model_size_mb():.4f} MB")

    # Run inference
    X = torch.tensor(test_features, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)
        _, predictions = torch.max(probs, dim=-1)

    y_true = test_labels
    y_pred = predictions.cpu().numpy()
    y_probs = probs.cpu().numpy()

    # ── Overall Accuracy ──
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'─' * 40}")
    print(f"Overall Accuracy:   {accuracy:.4f} ({accuracy:.1%})")
    print(f"F1 Score (macro):   {f1_macro:.4f}")
    print(f"F1 Score (weighted):{f1_weighted:.4f}")
    print(f"{'─' * 40}")

    # ── Classification Report ──
    report = classification_report(
        y_true, y_pred,
        target_names=config.GESTURE_CLASSES,
        digits=4,
    )
    report_dict = classification_report(
        y_true, y_pred,
        target_names=config.GESTURE_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)

    # Save classification report
    report_path = config.CLASSIFICATION_REPORT_PATH
    with open(report_path, "w") as f:
        f.write("Gesture Classifier — Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy:    {accuracy:.4f} ({accuracy:.1%})\n")
        f.write(f"F1 Score (macro):    {f1_macro:.4f}\n")
        f.write(f"F1 Score (weighted): {f1_weighted:.4f}\n")
        f.write(f"Model parameters:    {model.count_parameters():,}\n")
        f.write(f"Model size:          {model.model_size_mb():.4f} MB\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, config.GESTURE_CLASSES)

    # ── Per-class accuracy ──
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(config.GESTURE_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).mean()
            print(f"  {name:15s}: {class_acc:.4f} ({class_acc:.1%})  [{mask.sum()} samples]")

    # ── Average confidence analysis ──
    print("\nAverage Confidence Scores:")
    for i, name in enumerate(config.GESTURE_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            avg_conf = y_probs[mask, i].mean()
            print(f"  {name:15s}: {avg_conf:.4f}")

    # ── Save metrics as JSON ──
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "num_test_samples": int(len(test_features)),
        "model_params": model.count_parameters(),
        "model_size_mb": model.model_size_mb(),
        "class_names": config.GESTURE_CLASSES,
        "per_class_accuracy": {
            name: float((y_pred[y_true == i] == i).mean()) if (y_true == i).sum() > 0 else 0.0
            for i, name in enumerate(config.GESTURE_CLASSES)
        },
        "classification_report": report_dict,
    }
    with open(config.MODELS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {config.MODELS_DIR / 'evaluation_metrics.json'}")
    print("=" * 60)

    # Check if accuracy meets requirement
    if accuracy >= 0.85:
        print("✓ ACCURACY REQUIREMENT MET (>85%)")
    else:
        print("✗ WARNING: Accuracy below 85% target")


def _plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """Plot and save a beautiful confusion matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        square=True,
    )

    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Gesture Classification — Confusion Matrix (Normalized)", fontsize=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {config.CONFUSION_MATRIX_PATH}")

    # Also save raw counts
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        linewidths=0.5,
        square=True,
    )
    ax2.set_xlabel("Predicted Label", fontsize=13)
    ax2.set_ylabel("True Label", fontsize=13)
    ax2.set_title("Gesture Classification — Confusion Matrix (Raw Counts)", fontsize=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "confusion_matrix_counts.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    evaluate()
