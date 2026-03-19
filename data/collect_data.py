"""
Real Gesture Data Collector
=============================
Collects real hand-landmark samples from your webcam using MediaPipe.
Samples are saved to data/real_samples/ and can be used to fine-tune
or replace the synthetic training dataset.

Usage:
    python data/collect_data.py

Controls (in the OpenCV window):
    SPACE  → Save current frame's landmarks as a sample
    0-8    → Switch gesture class to collect
    q      → Quit

Gesture class indices:
    0=point_left  1=point_right  2=point_up    3=point_down
    4=open_palm   5=thumbs_up    6=thumbs_down 7=pinch  8=wave
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.hand_detector import HandDetector
from gestures.feature_extractor import FeatureExtractor

OUTPUT_DIR = Path(config.DATA_DIR) / "real_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def open_camera(index=0):
    for name, backend in [("AUTO", cv2.CAP_ANY), ("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"✓ Camera opened via {name}")
                return cap
            time.sleep(0.05)
        cap.release()
    return None


def main():
    print("=" * 60)
    print("Real Gesture Data Collector")
    print("=" * 60)
    print("Gesture classes:")
    for i, g in enumerate(config.GESTURE_CLASSES):
        print(f"  {i} → {g}")
    print()

    detector = HandDetector()
    extractor = FeatureExtractor()

    cap = open_camera(0)
    if cap is None:
        print("ERROR: Cannot open webcam.")
        return

    all_features = []
    all_labels = []
    current_class = 0
    counts = {i: 0 for i in range(config.NUM_CLASSES)}
    save_flash = 0  # Flash frames after save

    # Load existing samples if any
    if (OUTPUT_DIR / "features.npy").exists():
        all_features = list(np.load(str(OUTPUT_DIR / "features.npy")))
        all_labels = list(np.load(str(OUTPUT_DIR / "labels.npy")))
        for lbl in all_labels:
            counts[int(lbl)] += 1
        print(f"Loaded {len(all_labels)} existing samples.")

    print("\nPress SPACE to save a sample, 0-8 to select gesture class, q to quit")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        hands = detector.detect(frame)

        saved_this_frame = False
        for hand in hands:
            detector.draw_landmarks(display, hand, draw_bbox=True)

        # HUD
        gesture_name = config.GESTURE_CLASSES[current_class]
        color = (0, 255, 0) if save_flash > 0 else (255, 255, 255)
        cv2.putText(display, f"Gesture: [{current_class}] {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Samples: {counts[current_class]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(display, "SPACE=Save  0-8=Switch  q=Quit", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Count per class
        for i, g in enumerate(config.GESTURE_CLASSES):
            color_c = (0, 255, 0) if i == current_class else (160, 160, 160)
            cv2.putText(display, f"{i}:{g[:6]}={counts[i]}", (display.shape[1] - 165, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_c, 1)

        if save_flash > 0:
            cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), (0, 255, 0), 8)
            save_flash -= 1

        cv2.imshow("Collect Gesture Data", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif ord("0") <= key <= ord("8"):
            current_class = key - ord("0")
            print(f"Switched to class {current_class}: {config.GESTURE_CLASSES[current_class]}")
        elif key == ord(" ") and hands:
            hand = hands[0]  # Take first hand
            try:
                feat = extractor.extract(hand.landmarks, hand.handedness)
                all_features.append(feat)
                all_labels.append(current_class)
                counts[current_class] += 1
                save_flash = 5
                print(f"  Saved sample for '{gesture_name}' (total: {counts[current_class]})")
            except Exception as e:
                print(f"  Failed to extract features: {e}")

    # Save
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    if all_features:
        features_arr = np.array(all_features, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int64)
        np.save(str(OUTPUT_DIR / "features.npy"), features_arr)
        np.save(str(OUTPUT_DIR / "labels.npy"), labels_arr)

        metadata = {
            "num_samples": len(all_labels),
            "feature_dim": int(features_arr.shape[1]),
            "num_classes": config.NUM_CLASSES,
            "classes": config.GESTURE_CLASSES,
            "counts_per_class": {config.GESTURE_CLASSES[i]: counts[i] for i in range(config.NUM_CLASSES)},
        }
        with open(str(OUTPUT_DIR / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved {len(all_labels)} samples to {OUTPUT_DIR}")
        print("Per-class counts:")
        for i, g in enumerate(config.GESTURE_CLASSES):
            print(f"  {g}: {counts[i]}")
        print("\nTo retrain with real data:")
        print("  python training/train.py --data data/real_samples/")


if __name__ == "__main__":
    main()
