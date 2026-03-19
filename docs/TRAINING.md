# Training Documentation

## Overview

The gesture classifier is trained on synthetically generated hand landmark data with geometric augmentations. This approach creates a self-contained, reproducible training pipeline that doesn't require external dataset downloads.

## Data Generation Strategy

### Why Synthetic Data?

| Approach | Pros | Cons |
|----------|------|------|
| External dataset (HAGRID) | Real-world variety | Multi-GB download, labeling format mismatch |
| Webcam collection | Exact domain match | Time-consuming, limited variety |
| **Synthetic generation** | **Fast, reproducible, self-contained** | **Requires anatomical accuracy** |

We chose synthetic generation because our classifier operates on **landmark features** (not raw pixels). The landmarks come from MediaPipe's pre-trained model, which already handles visual complexity. We only need to generate realistic landmark configurations for each gesture.

### Template Design

Each gesture has a hand-crafted "ideal" landmark template based on hand anatomy:

- **Point directions**: Index finger extended in target direction, other fingers curled
- **Open palm**: All 5 fingers fully extended and spread
- **Thumbs up/down**: Thumb extended vertically, other 4 fingers in fist
- **Pinch**: Thumb tip and index tip brought close together
- **Wave**: Same landmarks as open_palm (detected temporally)

### Augmentation Pipeline

Each template is augmented to create diverse samples:

1. **Gaussian Noise** (σ=0.02): Simulates landmark detection jitter
2. **Random Scale** (0.8×–1.2×): Simulates varying hand-camera distance
3. **Random Translation** (±0.15): Simulates hand position in frame
4. **Random 2D Rotation** (±15°): Simulates wrist rotation

**Samples per class**: 800 (configurable via `--samples` flag)  
**Total dataset**: ~7,200 samples

## Training Process

### Split Strategy

| Split | Ratio | Purpose |
|-------|-------|---------|
| Training | 70% | Model weight updates |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (never seen during training) |

### Optimizer: Adam

- Learning rate: 1e-3
- Weight decay: 1e-4 (L2 regularization)
- Chosen for: fast convergence, adaptive per-parameter learning rates

### Learning Rate Schedule: ReduceLROnPlateau

- Monitors: validation loss
- Factor: 0.5 (halves LR when plateau detected)
- Patience: 10 epochs
- Benefit: automatically finds optimal learning rate stages

### Early Stopping

- Monitors: validation loss
- Patience: 15 epochs
- Saves: best model checkpoint based on lowest validation loss
- Prevents overfitting without manual epoch tuning

### Loss Function: Cross-Entropy Loss

- Standard choice for multi-class classification
- Naturally handles class probabilities via log-softmax
- No class weighting needed (balanced dataset by construction)

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden dim 1 | 128 | Sufficient capacity for 78→9 mapping |
| Hidden dim 2 | 64 | Gradual bottleneck towards output |
| Dropout | 0.3 | Regularization without excessive capacity loss |
| Batch size | 64 | Good GPU utilization, stable gradients |
| Max epochs | 100 | Upper bound (early stopping typically triggers earlier) |
| EMA α | 0.6 | Balance responsiveness and stability |

## Running Training

```bash
# Step 1: Generate dataset
python data/generate_dataset.py --samples 800

# Step 2: Train on synthetic data
python training/train.py --epochs 100 --lr 0.001

# Or train on collected webcam features
python training/train.py --data data/real_samples --epochs 100 --lr 0.001

# Step 3: Evaluate
python training/evaluate.py

# Step 4: Export to ONNX
python training/export_onnx.py

# Step 5: Benchmark runtime
python training/benchmark.py --seconds 15
```

### Custom Training Options

```bash
# More samples for better generalization
python data/generate_dataset.py --samples 2000

# Longer training with lower learning rate
python training/train.py --epochs 200 --lr 0.0005 --batch-size 128
```

## Output Artifacts

After training, the `models/` directory contains:

| File | Description |
|------|-------------|
| `gesture_classifier_best.pth` | Best PyTorch checkpoint (weights + metadata) |
| `gesture_classifier.onnx` | ONNX exported model |
| `training_curves.png` | Loss and accuracy curves |
| `confusion_matrix.png` | Normalized confusion matrix |
| `confusion_matrix_counts.png` | Raw count confusion matrix |
| `classification_report.txt` | Per-class precision/recall/F1 |
| `evaluation_metrics.json` | Machine-readable metrics |
| `training_history.json` | Per-epoch loss/accuracy/LR |
| `runtime_metrics.json` | Webcam FPS/latency benchmark |
| `dataset_summary.json` | Dataset source and split metadata |

## Transfer Learning & Fine-Tuning

To improve real-world performance:

1. **Collect real data**: Use `data/collect_data.py` to capture webcam samples
2. **Combine datasets**: Merge synthetic + real data
3. **Fine-tune**: Load pre-trained weights and continue training with lower LR

```python
# Example fine-tuning snippet
checkpoint = torch.load("models/gesture_classifier_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])
# Train with lower learning rate (0.0001) on combined data
```
