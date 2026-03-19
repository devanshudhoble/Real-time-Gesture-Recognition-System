"""
Model Training Script
======================
Trains the gesture classification MLP on the synthetic dataset.

Features:
    - Train/val/test splits (70/15/15)
    - Adam optimizer with weight decay
    - Learning rate scheduling (ReduceLROnPlateau)
    - Early stopping to prevent overfitting
    - Saves best model checkpoint
    - Logs training/validation loss and accuracy curves
    - Saves training history as JSON for analysis

Usage:
    python training/train.py
    python training/train.py --epochs 200 --lr 0.0005
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.classifier import GestureClassifier


def load_dataset(data_dir: Path):
    """Load a feature dataset from disk."""
    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.npy"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Dataset not found in '{data_dir}'. Expected files:\n"
            f"  - {features_path}\n"
            f"  - {labels_path}"
        )

    features = np.load(features_path)
    labels = np.load(labels_path)
    print(
        f"Loaded dataset from {data_dir}: "
        f"{features.shape[0]} samples, {features.shape[1]} features"
    )
    return features, labels


def split_dataset(features: np.ndarray, labels: np.ndarray, seed: int):
    """Split into stratified train/val/test sets."""
    test_ratio = config.TEST_SPLIT
    val_ratio = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val,
    )

    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    )


def create_dataloaders(train_data, val_data, batch_size):
    """Create PyTorch DataLoaders."""
    train_ds = TensorDataset(
        torch.tensor(train_data[0], dtype=torch.float32),
        torch.tensor(train_data[1], dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(val_data[0], dtype=torch.float32),
        torch.tensor(val_data[1], dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Gesture Classifier Training")
    print("=" * 60)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and split data
    data_dir = Path(args.data).resolve() if args.data else config.DATA_DIR
    features, labels = load_dataset(data_dir)
    train_data, val_data, test_data = split_dataset(features, labels, args.seed)

    print(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")

    # Save test set for evaluation
    config.MODELS_DIR.mkdir(exist_ok=True)
    np.save(config.MODELS_DIR / "test_features.npy", test_data[0])
    np.save(config.MODELS_DIR / "test_labels.npy", test_data[1])
    _save_dataset_summary(
        data_dir=data_dir,
        features=features,
        labels=labels,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        seed=args.seed,
    )

    # Create model
    input_dim = features.shape[1]
    model = GestureClassifier(input_dim=input_dim, num_classes=config.NUM_CLASSES)
    model.to(device)

    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Model size: {model.model_size_mb():.4f} MB")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.LR_PATIENCE, factor=0.5,
    )

    # DataLoaders
    train_loader, val_loader = create_dataloaders(train_data, val_data, args.batch_size)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(features_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels_batch)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels_batch).sum().item()
            train_total += len(labels_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)

                logits = model(features_batch)
                loss = criterion(logits, labels_batch)

                val_loss += loss.item() * len(labels_batch)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels_batch).sum().item()
                val_total += len(labels_batch)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} │ "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} │ "
                f"LR: {current_lr:.6f}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "input_dim": input_dim,
                "num_classes": config.NUM_CLASSES,
                "gesture_classes": config.GESTURE_CLASSES,
            }
            torch.save(checkpoint, config.BEST_MODEL_PATH)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={config.EARLY_STOP_PATIENCE})")
            break

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")

    # Save training history
    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    _plot_training_curves(history)

    # Print model size
    model_size = config.BEST_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Model file size: {model_size:.4f} MB (requirement: <500 MB ✓)")
    print("=" * 60)


def _plot_training_curves(history: dict):
    """Plot and save training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {config.MODELS_DIR / 'training_curves.png'}")


def _class_distribution(labels: np.ndarray) -> dict:
    """Return class counts keyed by gesture name."""
    counts = {}
    for idx, name in enumerate(config.GESTURE_CLASSES):
        counts[name] = int((labels == idx).sum())
    return counts


def _save_dataset_summary(
    data_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    train_data,
    val_data,
    test_data,
    seed: int,
):
    """Persist training split metadata for submission artifacts."""
    summary = {
        "dataset_dir": str(data_dir),
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "seed": int(seed),
        "splits": {
            "train": int(len(train_data[0])),
            "val": int(len(val_data[0])),
            "test": int(len(test_data[0])),
        },
        "class_distribution": {
            "full": _class_distribution(labels),
            "train": _class_distribution(train_data[1]),
            "val": _class_distribution(val_data[1]),
            "test": _class_distribution(test_data[1]),
        },
    }
    with open(config.MODELS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Dataset summary saved to: {config.MODELS_DIR / 'dataset_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Directory containing features.npy and labels.npy (default: data/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splits",
    )
    args = parser.parse_args()
    train(args)
