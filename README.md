# 🤚 Real-Time Gesture Recognition System

A production-ready, real-time hand gesture recognition system that detects and classifies **9 hand gestures** from webcam input at **20+ FPS on CPU**.

Built with **MediaPipe** (hand detection), **PyTorch** (classification), and **OpenCV** (video processing), with **ONNX Runtime** support for optimized edge deployment.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Supported Gestures

| # | Gesture | Description | Use Case |
|---|---------|-------------|----------|
| 0 | `point_left` | Index finger pointing left | Navigation |
| 1 | `point_right` | Index finger pointing right | Navigation |
| 2 | `point_up` | Index finger pointing up | Scroll / Select |
| 3 | `point_down` | Index finger pointing down | Scroll / Select |
| 4 | `open_palm` | All fingers spread | Stop / Pause |
| 5 | `thumbs_up` | Thumb up, fist closed | Confirm / Like |
| 6 | `thumbs_down` | Thumb down, fist closed | Reject / Dislike |
| 7 | `pinch` | Thumb + index close together | Zoom / Precision |
| 8 | `wave` | Open palm oscillating | Greeting / Attention |

---

## 🏗️ Architecture Overview

```
VideoCapture → PreProcessing → HandDetection → FeatureExtraction → GestureClassification → PostProcessing → Output
     │              │                │                  │                      │                    │            │
   OpenCV      BGR→RGB          MediaPipe          78-dim vector         PyTorch MLP        EMA + Hyster.   Label +
              + flip          (21 landmarks)      (normalized)          (128→64→9)        + Wave detect   Confidence
```

### Pipeline Components

| Component | Technology | Latency | Purpose |
|-----------|-----------|---------|---------|
| Hand Detection | MediaPipe Hands | ~15ms | Locate hands, extract 21 3D landmarks |
| Feature Extraction | NumPy | <1ms | Convert landmarks → 78-dim invariant vector |
| Classification | PyTorch MLP | <1ms | Predict gesture from feature vector |
| Post-Processing | Custom | <1ms | Temporal smoothing, confidence filtering |
| **Total** | | **<30ms** | **End-to-end latency** |

### Why MLP over CNN / Vision Transformer?

Our input is a **78-dimensional structured feature vector** (not raw pixels):
- **CNNs** excel at spatial hierarchies in images — not applicable here
- **Vision Transformers** shine with patch-based attention on images — overkill for 78 floats
- **MLP** achieves >95% accuracy with **<1MB model size** and **<1ms inference** — ideal for real-time edge deployment

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gesture-recognition.git
cd gesture-recognition

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python data/generate_dataset.py
```

This creates ~7,200 synthetic samples (800 per gesture × 9 classes) using anatomically-accurate landmark templates with augmentation (noise, rotation, scale, translation).

### 3. Train the Model

```bash
python training/train.py
python training/train.py --data data/real_samples
```

Training runs with:
- Adam optimizer + weight decay (1e-4)
- ReduceLROnPlateau learning rate scheduler
- Early stopping (patience=15)
- Saves best model to `models/gesture_classifier_best.pth`

### 4. Evaluate

```bash
python training/evaluate.py
```

Generates:
- Classification report (precision, recall, F1 per class)
- Confusion matrix (saved to `models/confusion_matrix.png`)
- Per-class accuracy and confidence analysis

### 5. Export to ONNX (Optional)

```bash
python training/export_onnx.py
```

Exports to ONNX format with validation and PyTorch vs ONNX Runtime speed comparison.

### 6. Benchmark Real-Time Performance

```bash
python training/benchmark.py
python training/benchmark.py --onnx --seconds 20
```

Saves runtime metrics to `models/runtime_metrics.json`.

### 7. Run Live Demo

```bash
python demo/demo.py             # PyTorch backend
python demo/demo.py --onnx      # ONNX Runtime (faster)
python demo/demo.py --camera 1  # Use different camera
```

On macOS, the default camera preference is the built-in MacBook camera. If
Continuity Camera or another device should be used instead, pass `--camera`
explicitly or set `GESTURE_CAMERA_INDEX`.

**Demo controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Take screenshot |
| `r` | Reset smoothing |
| `f` | Toggle FPS display |
| `h` | Toggle help overlay |

### 8. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >85% | >95% (synthetic test set) |
| FPS (CPU) | >20 | ~25-30 FPS |
| Latency | <100ms | <30ms end-to-end |
| Model Size | <500MB | <0.1 MB |
| Memory | Moderate | ~200MB runtime |

---

## 📁 Project Structure

```
gesture-recognition/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── config.py                      # Central configuration
│
├── gestures/                      # Core library
│   ├── __init__.py
│   ├── hand_detector.py           # MediaPipe hand detection wrapper
│   ├── feature_extractor.py       # Landmark → 78-dim feature vector
│   ├── classifier.py              # PyTorch MLP (128→64→9)
│   ├── pipeline.py                # End-to-end inference orchestrator
│   ├── post_processor.py          # EMA smoothing, hysteresis, wave detection
│   └── utils.py                   # FPS counter, latency tracker, drawing helpers
│
├── data/
│   ├── generate_dataset.py        # Synthetic dataset generator
│   └── collect_data.py            # Interactive webcam data collection
│
├── training/
│   ├── train.py                   # Training with early stopping & LR scheduling
│   ├── evaluate.py                # Evaluation, confusion matrix, metrics
│   ├── export_onnx.py             # ONNX export + backend validation
│   └── benchmark.py               # Runtime FPS / latency benchmark
│
├── demo/
│   └── demo.py                    # Real-time webcam demo application
│
├── models/                        # Generated artifacts
│   ├── gesture_classifier_best.pth
│   ├── gesture_classifier.onnx
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.json
│   ├── runtime_metrics.json
│   └── dataset_summary.json
│
├── docs/
│   ├── ARCHITECTURE.md            # Detailed architecture explanation
│   ├── TRAINING.md                # Training process documentation
│   ├── PERFORMANCE.md             # Performance analysis
│   └── SUBMISSION.md              # Requirement-to-deliverable mapping
│
└── tests/
    ├── test_feature_extractor.py  # Feature extraction tests
    ├── test_classifier.py         # Model architecture tests
    └── test_pipeline.py           # Post-processor & integration tests
```

---

## 🔧 Technical Details

### Feature Engineering (78 dimensions)

| Feature Group | Count | Description |
|--------------|-------|-------------|
| Relative positions | 60 | Each landmark (x,y,z) relative to wrist, normalized by palm size |
| Extension ratios | 5 | Finger tip-to-MCP / PIP-to-MCP distance ratios |
| Inter-finger distances | 10 | Pairwise tip distances (C(5,2) combinations) |
| Thumb-index distance | 1 | Key feature for pinch detection |
| Palm orientation | 2 | XY-plane angle + Z-tilt from palm vector |

### Temporal Post-Processing

1. **EMA Smoothing** (α=0.6): Exponential moving average on softmax probabilities across frames
2. **Confidence Thresholding** (≥55%): Reject low-confidence predictions
3. **Hysteresis** (3 frames): Require 3 consecutive agreeing frames before switching gesture
4. **Wave Detection**: Temporal oscillation analysis of wrist x-coordinate over 15-frame window

### Edge Case Handling

- **No hands detected**: Returns empty results, no crash
- **Partial occlusion**: Graceful degradation via try/except in pipeline
- **Motion blur**: MediaPipe tracking handles inter-frame motion; EMA smooths classifier noise
- **Varying lighting**: MediaPipe's ML-based detection is robust to lighting; our features use geometric ratios, not pixel intensities
- **Left/Right hands**: Left hands are mirrored to right-hand coordinate space before feature extraction

---

## 🔮 Future Improvements

- **Real-world dataset**: Fine-tune on actual webcam captures (use `data/collect_data.py`)
- **Gesture sequences**: Implement HMM or LSTM for recognizing swipe patterns and sequences
- **3D joint pose estimation**: Export full 3D skeleton for AR/VR applications
- **Two-handed gestures**: Combine features from both hands for cooperative gestures
- **Custom gesture training**: User-defined gestures via few-shot learning
- **Mobile deployment**: TensorFlow Lite / Core ML export for iOS/Android
- **WebAssembly**: Browser-based demo using TF.js or ONNX Web
- **Multi-camera**: Fuse detections from multiple viewpoints for 3D reconstruction

---

## 📚 References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) — Google's hand landmark detection
- [ONNX Runtime](https://onnxruntime.ai/) — High-performance ML inference
- [HAGRID Dataset](https://github.com/hukenovs/hagrid) — Hand gesture recognition dataset
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Google MediaPipe team for the hand landmark model
- PyTorch team for the deep learning framework
- OpenCV community for computer vision utilities
