# Architecture Documentation

## System Overview

The gesture recognition system follows a modular, pipeline-based architecture optimized for real-time CPU inference. Each component is independently testable and replaceable.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Gesture Recognition Pipeline                      │
│                                                                       │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  OpenCV   │───►│ MediaPipe │───►│ Feature  │───►│   PyTorch    │  │
│  │ VideoCapt │    │   Hands   │    │Extraction│    │  MLP (3-lyr) │  │
│  └──────────┘    └───────────┘    └──────────┘    └──────────────┘  │
│       │                │                │                │           │
│   BGR Frame      21 Landmarks     78-dim Vector     9-class Probs   │
│                  per hand         (normalized)      (softmax)       │
│                                                          │           │
│                                              ┌──────────────────┐   │
│                                              │  Post-Processor   │   │
│                                              │  EMA + Hysteresis │   │
│                                              │  + Wave Detection │   │
│                                              └──────────────────┘   │
│                                                          │           │
│                                                  Gesture + Conf.    │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Hand Detector (`gestures/hand_detector.py`)

**Technology**: MediaPipe Hands (Google)

**Responsibilities**:
- Accept BGR video frames from OpenCV
- Detect up to 2 hands simultaneously
- Return 21 3D landmarks + handedness + bounding box per hand
- Handle varying lighting via configurable confidence thresholds

**Why MediaPipe?**
- Pre-trained, production-quality ML model
- Runs efficiently on CPU (~15ms per frame)
- Works across skin tones and lighting conditions
- Provides both normalized (image-relative) and world (metric) landmarks
- Used by Google in production applications

**Configuration**:
- `max_num_hands`: 2
- `model_complexity`: 1 (full model for accuracy)
- `min_detection_confidence`: 0.5
- `min_tracking_confidence`: 0.5

### 2. Feature Extractor (`gestures/feature_extractor.py`)

**Technology**: NumPy (pure computation)

**Input**: 21 × 3 normalized landmarks from MediaPipe  
**Output**: 78-dimensional feature vector

**Feature Groups**:

| Group | Features | Count | Purpose |
|-------|----------|-------|---------|
| Relative Positions | Landmarks 1-20 relative to wrist, normalized by palm size | 60 | Capture hand pose geometry |
| Extension Ratios | tip-MCP / PIP-MCP distance per finger | 5 | Finger open vs curled |
| Inter-Finger Distances | Pairwise tip-to-tip distances | 10 | Finger spread pattern |
| Thumb-Index Distance | Distance between thumb and index tips | 1 | Pinch detection |
| Palm Orientation | XY angle + Z tilt of palm vector | 2 | Hand orientation |

**Invariance Properties**:
- **Translation invariant**: All features are relative to wrist position
- **Scale invariant**: Normalized by palm size (wrist-to-middle-MCP distance)
- **Hand agnostic**: Left hands are mirrored to right-hand coordinates

### 3. Gesture Classifier (`gestures/classifier.py`)

**Technology**: PyTorch (nn.Module)

**Architecture**:
```
Input(78) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(9)   [logits]
```

**Design Rationale**:

| Approach | Input Type | Model Size | Inference Time | Our Case |
|----------|-----------|------------|----------------|----------|
| CNN (e.g., ResNet) | Raw images (H×W×3) | 10-100+ MB | 10-50ms | ❌ Overkill for 78 floats |
| Vision Transformer | Image patches | 50-300+ MB | 20-100ms | ❌ Patch attention not applicable |
| **MLP** | **Feature vector (78)** | **<0.1 MB** | **<1ms** | **✅ Optimal fit** |
| Random Forest | Feature vector | N/A | ~1ms | ❌ Less smooth probability outputs |

**Key properties**:
- BatchNorm for training stability
- Dropout (0.3) for regularization
- Xavier weight initialization
- ~15,000 parameters, <0.1 MB

### 4. Post-Processor (`gestures/post_processor.py`)

**Technology**: Custom NumPy-based temporal logic

**Techniques**:

1. **Exponential Moving Average (EMA)**
   - Formula: `smoothed[t] = α × raw[t] + (1-α) × smoothed[t-1]`
   - α = 0.6 (responsive but stable)
   - Purpose: Remove jittery frame-to-frame noise

2. **Confidence Thresholding**
   - Minimum: 55%
   - Below threshold → report "no gesture" instead of wrong prediction

3. **Hysteresis (Label Stability)**
   - Requires 3 consecutive frames predicting the same new gesture
   - Prevents rapid label flickering between similar gestures

4. **Wave Detection**
   - Tracks wrist x-coordinate over 15-frame sliding window
   - Counts direction reversals (left↔right oscillations)
   - Requires ≥2 reversals + open palm → classified as "wave"
   - Purely temporal — cannot be detected from a single frame

### 5. Pipeline Orchestrator (`gestures/pipeline.py`)

**Responsibilities**:
- Chain all components in correct order
- Handle multiple hands per frame
- Graceful error handling for partial occlusion
- Support both PyTorch and ONNX Runtime backends
- Track FPS and latency statistics

### 6. Utilities (`gestures/utils.py`)

- **FPSCounter**: Sliding-window timestamp-based FPS calculation
- **LatencyTracker**: Per-frame latency with mean/p50/p95/p99 statistics
- **draw_text_with_bg**: Text overlay with background for readability
- **draw_confidence_bar**: Color-coded confidence bar visualization

## Data Flow

```
Frame (640×480×3 BGR)
    │
    ▼
HandDetector.detect()
    │ → List[HandResult] (landmarks, handedness, bbox)
    │
    ▼ (for each hand)
FeatureExtractor.extract()
    │ → np.ndarray (78,)
    │
    ▼
GestureClassifier.predict_proba()
    │ → np.ndarray (9,) softmax probabilities
    │
    ▼
PostProcessor.process()
    │ → (gesture_name, confidence, smoothed_probs)
    │
    ▼
GestureResult (gesture, confidence, landmarks, bbox, handedness)
```

## Edge Case Handling Strategy

| Edge Case | Strategy |
|-----------|----------|
| No hands visible | Return empty results list |
| Partial occlusion | try/except wraps feature extraction; skip frame silently |
| Motion blur | MediaPipe tracking provides temporal consistency; EMA smooths classifier output |
| Varying lighting | MediaPipe is ML-based (not rule-based); features use geometric ratios, not intensities |
| Hand leaves/enters frame | PostProcessor.reset() when hand tracking lost |
| Very fast gesture transitions | Hysteresis prevents premature switches; 3-frame confirmation |
| Ambiguous gestures | Confidence threshold rejects low-confidence; top-3 shown for transparency |
