# Submission Checklist

This repository is structured to match the assignment deliverables exactly.

## Requirement Mapping

| Assignment Requirement | Repository Implementation |
|------------------------|---------------------------|
| Hand detection | `gestures/hand_detector.py` using MediaPipe Tasks |
| Gesture classification | `gestures/classifier.py` over 9 gesture classes |
| Real-time webcam processing | `demo/demo.py` and `gestures/pipeline.py` |
| Confidence scoring | `gestures/post_processor.py` returns smoothed confidence |
| Multiple hands | MediaPipe configured for up to 2 hands |
| Smooth tracking | EMA smoothing + hysteresis + MediaPipe tracking |
| Training script | `training/train.py` |
| Inference pipeline | `gestures/pipeline.py` |
| Demo application | `demo/demo.py` |
| Performance metrics | `training/benchmark.py` → `models/runtime_metrics.json` |
| Confusion matrix | `training/evaluate.py` → `models/confusion_matrix.png` |
| Model artifacts | `models/gesture_classifier_best.pth`, `models/gesture_classifier.onnx` |
| Documentation | `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/PERFORMANCE.md` |

## Submission Commands

Run these commands from the project root after installing dependencies.

```bash
# 1. Create training data
python data/generate_dataset.py

# Optional: collect real webcam samples instead
python data/collect_data.py

# 2. Train
python training/train.py
# or
python training/train.py --data data/real_samples

# 3. Evaluate
python training/evaluate.py

# 4. Export optimized model
python training/export_onnx.py

# 5. Benchmark runtime
python training/benchmark.py --seconds 15

# 6. Run live demo
python demo/demo.py
```

## Expected Artifacts

After running the pipeline, the `models/` directory should contain:

- `gesture_classifier_best.pth`
- `gesture_classifier.onnx`
- `training_curves.png`
- `training_history.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `confusion_matrix_counts.png`
- `evaluation_metrics.json`
- `runtime_metrics.json`
- `dataset_summary.json`

## Notes For Reviewers

- The system uses landmark-based recognition rather than raw-image classification.
- `wave` is handled as a temporal gesture in post-processing, not as a static pose alone.
- For the strongest real-world submission, collect a small real webcam dataset and train with `--data data/real_samples`.
- On macOS, the default camera preference is the built-in MacBook camera; override with `--camera` or `GESTURE_CAMERA_INDEX` if needed.
