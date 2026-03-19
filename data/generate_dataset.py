"""
Synthetic Dataset Generator
============================
Generates synthetic hand-landmark feature vectors for training the gesture
classifier without requiring external dataset downloads.

Strategy:
    For each gesture class, we define an "ideal" hand landmark configuration
    based on anatomical knowledge of that gesture. We then augment each ideal
    pose with random noise, rotation, and scale perturbations to create a
    diverse, realistic training set.

    This approach is effective because our classifier operates on landmark-
    derived features (not raw pixels), so we only need realistic landmark
    distributions — not photorealistic images.

Usage:
    python data/generate_dataset.py              # Generate and save dataset
    python data/generate_dataset.py --samples 1000  # Custom sample count
"""

import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from gestures.feature_extractor import FeatureExtractor


# ═══════════════════════════════════════════════
# Ideal Landmark Templates
# ═══════════════════════════════════════════════
# Each template is a (21, 3) array representing the normalized landmark
# positions for an ideal instance of the gesture. These are defined in
# a coordinate system where:
#   x: horizontal (0=left, 1=right)
#   y: vertical   (0=top, 1=bottom)
#   z: depth      (negative = towards camera)

def _base_hand() -> np.ndarray:
    """Return a neutral 'rest' hand configuration (fist-like)."""
    landmarks = np.array([
        # Wrist
        [0.50, 0.80, 0.00],
        # Thumb: CMC, MCP, IP, TIP
        [0.42, 0.72, -0.02],
        [0.35, 0.65, -0.03],
        [0.32, 0.60, -0.03],
        [0.30, 0.56, -0.03],
        # Index: MCP, PIP, DIP, TIP
        [0.42, 0.58, -0.02],
        [0.40, 0.50, -0.02],
        [0.41, 0.45, -0.01],
        [0.42, 0.42, -0.01],
        # Middle: MCP, PIP, DIP, TIP
        [0.48, 0.56, -0.02],
        [0.48, 0.48, -0.02],
        [0.48, 0.43, -0.01],
        [0.48, 0.40, -0.01],
        # Ring: MCP, PIP, DIP, TIP
        [0.54, 0.58, -0.02],
        [0.54, 0.50, -0.02],
        [0.54, 0.46, -0.01],
        [0.55, 0.43, -0.01],
        # Pinky: MCP, PIP, DIP, TIP
        [0.60, 0.62, -0.01],
        [0.61, 0.55, -0.01],
        [0.62, 0.51, -0.01],
        [0.62, 0.48, -0.01],
    ], dtype=np.float32)
    return landmarks


def _open_palm() -> np.ndarray:
    """All fingers fully extended and spread — stop gesture."""
    lm = np.array([
        [0.50, 0.85, 0.00],   # Wrist
        [0.38, 0.75, -0.02],  # Thumb CMC
        [0.30, 0.65, -0.03],  # Thumb MCP
        [0.25, 0.55, -0.04],  # Thumb IP
        [0.20, 0.48, -0.04],  # Thumb TIP
        [0.38, 0.58, -0.02],  # Index MCP
        [0.36, 0.45, -0.02],  # Index PIP
        [0.35, 0.35, -0.02],  # Index DIP
        [0.34, 0.28, -0.02],  # Index TIP
        [0.46, 0.55, -0.02],  # Middle MCP
        [0.46, 0.42, -0.02],  # Middle PIP
        [0.46, 0.32, -0.02],  # Middle DIP
        [0.46, 0.25, -0.02],  # Middle TIP
        [0.54, 0.57, -0.02],  # Ring MCP
        [0.55, 0.44, -0.02],  # Ring PIP
        [0.56, 0.34, -0.02],  # Ring DIP
        [0.57, 0.28, -0.02],  # Ring TIP
        [0.62, 0.62, -0.01],  # Pinky MCP
        [0.64, 0.50, -0.01],  # Pinky PIP
        [0.66, 0.42, -0.01],  # Pinky DIP
        [0.67, 0.36, -0.01],  # Pinky TIP
    ], dtype=np.float32)
    return lm


def _point_up() -> np.ndarray:
    """Index finger pointing up, other fingers curled."""
    lm = _base_hand().copy()
    # Curl all fingers except index
    # Thumb curled inward
    lm[3] = [0.38, 0.63, -0.02]
    lm[4] = [0.40, 0.60, -0.01]
    # Index extended upward
    lm[6] = [0.42, 0.45, -0.02]
    lm[7] = [0.42, 0.35, -0.02]
    lm[8] = [0.42, 0.25, -0.02]
    # Middle curled
    lm[10] = [0.48, 0.55, -0.02]
    lm[11] = [0.49, 0.58, -0.01]
    lm[12] = [0.48, 0.60, 0.00]
    # Ring curled
    lm[14] = [0.54, 0.57, -0.02]
    lm[15] = [0.55, 0.60, -0.01]
    lm[16] = [0.54, 0.62, 0.00]
    # Pinky curled
    lm[18] = [0.60, 0.62, -0.01]
    lm[19] = [0.61, 0.65, 0.00]
    lm[20] = [0.60, 0.67, 0.00]
    return lm


def _point_down() -> np.ndarray:
    """Index finger pointing down, other fingers curled."""
    lm = _base_hand().copy()
    lm[3] = [0.38, 0.63, -0.02]
    lm[4] = [0.40, 0.60, -0.01]
    # Index extended downward
    lm[6] = [0.42, 0.65, -0.02]
    lm[7] = [0.42, 0.75, -0.02]
    lm[8] = [0.42, 0.85, -0.02]
    # Middle curled
    lm[10] = [0.48, 0.55, -0.02]
    lm[11] = [0.49, 0.58, -0.01]
    lm[12] = [0.48, 0.60, 0.00]
    # Ring curled
    lm[14] = [0.54, 0.57, -0.02]
    lm[15] = [0.55, 0.60, -0.01]
    lm[16] = [0.54, 0.62, 0.00]
    # Pinky curled
    lm[18] = [0.60, 0.62, -0.01]
    lm[19] = [0.61, 0.65, 0.00]
    lm[20] = [0.60, 0.67, 0.00]
    return lm


def _point_left() -> np.ndarray:
    """Index finger pointing left, other fingers curled."""
    lm = _base_hand().copy()
    lm[3] = [0.38, 0.63, -0.02]
    lm[4] = [0.40, 0.60, -0.01]
    # Index extended leftward
    lm[6] = [0.35, 0.55, -0.02]
    lm[7] = [0.25, 0.55, -0.02]
    lm[8] = [0.15, 0.55, -0.02]
    # Middle curled
    lm[10] = [0.48, 0.55, -0.02]
    lm[11] = [0.49, 0.58, -0.01]
    lm[12] = [0.48, 0.60, 0.00]
    # Ring curled
    lm[14] = [0.54, 0.57, -0.02]
    lm[15] = [0.55, 0.60, -0.01]
    lm[16] = [0.54, 0.62, 0.00]
    # Pinky curled
    lm[18] = [0.60, 0.62, -0.01]
    lm[19] = [0.61, 0.65, 0.00]
    lm[20] = [0.60, 0.67, 0.00]
    return lm


def _point_right() -> np.ndarray:
    """Index finger pointing right, other fingers curled."""
    lm = _base_hand().copy()
    lm[3] = [0.38, 0.63, -0.02]
    lm[4] = [0.40, 0.60, -0.01]
    # Index extended rightward
    lm[6] = [0.55, 0.55, -0.02]
    lm[7] = [0.65, 0.55, -0.02]
    lm[8] = [0.78, 0.55, -0.02]
    # Middle curled
    lm[10] = [0.48, 0.55, -0.02]
    lm[11] = [0.49, 0.58, -0.01]
    lm[12] = [0.48, 0.60, 0.00]
    # Ring curled
    lm[14] = [0.54, 0.57, -0.02]
    lm[15] = [0.55, 0.60, -0.01]
    lm[16] = [0.54, 0.62, 0.00]
    # Pinky curled
    lm[18] = [0.60, 0.62, -0.01]
    lm[19] = [0.61, 0.65, 0.00]
    lm[20] = [0.60, 0.67, 0.00]
    return lm


def _thumbs_up() -> np.ndarray:
    """Thumb extended upward, all other fingers curled into fist."""
    lm = _base_hand().copy()
    # Thumb pointing up
    lm[1] = [0.42, 0.70, -0.02]
    lm[2] = [0.38, 0.58, -0.04]
    lm[3] = [0.36, 0.48, -0.05]
    lm[4] = [0.35, 0.38, -0.05]
    # Index curled
    lm[6] = [0.42, 0.55, -0.02]
    lm[7] = [0.44, 0.60, -0.01]
    lm[8] = [0.43, 0.63, 0.00]
    # Middle curled
    lm[10] = [0.48, 0.55, -0.02]
    lm[11] = [0.50, 0.60, -0.01]
    lm[12] = [0.49, 0.63, 0.00]
    # Ring curled
    lm[14] = [0.54, 0.57, -0.02]
    lm[15] = [0.55, 0.62, -0.01]
    lm[16] = [0.54, 0.64, 0.00]
    # Pinky curled
    lm[18] = [0.60, 0.62, -0.01]
    lm[19] = [0.61, 0.66, 0.00]
    lm[20] = [0.60, 0.68, 0.00]
    return lm


def _thumbs_down() -> np.ndarray:
    """Thumb extended downward, all other fingers curled."""
    lm = _base_hand().copy()
    # Thumb pointing down
    lm[1] = [0.42, 0.72, -0.02]
    lm[2] = [0.38, 0.80, -0.04]
    lm[3] = [0.36, 0.88, -0.05]
    lm[4] = [0.35, 0.95, -0.05]
    # Index curled
    lm[6] = [0.42, 0.55, -0.02]
    lm[7] = [0.44, 0.60, -0.01]
    lm[8] = [0.43, 0.63, 0.00]
    # Middle curled
    lm[10] = [0.48, 0.55, -0.02]
    lm[11] = [0.50, 0.60, -0.01]
    lm[12] = [0.49, 0.63, 0.00]
    # Ring curled
    lm[14] = [0.54, 0.57, -0.02]
    lm[15] = [0.55, 0.62, -0.01]
    lm[16] = [0.54, 0.64, 0.00]
    # Pinky curled
    lm[18] = [0.60, 0.62, -0.01]
    lm[19] = [0.61, 0.66, 0.00]
    lm[20] = [0.60, 0.68, 0.00]
    return lm


def _pinch() -> np.ndarray:
    """Thumb and index finger tips close together (pinch/zoom)."""
    lm = _base_hand().copy()
    # Thumb reaching toward index
    lm[1] = [0.42, 0.72, -0.02]
    lm[2] = [0.38, 0.62, -0.03]
    lm[3] = [0.40, 0.55, -0.04]
    lm[4] = [0.43, 0.50, -0.04]  # Thumb tip
    # Index reaching toward thumb
    lm[6] = [0.42, 0.58, -0.02]
    lm[7] = [0.42, 0.52, -0.02]
    lm[8] = [0.43, 0.50, -0.03]  # Index tip — very close to thumb tip
    # Middle extended
    lm[10] = [0.48, 0.55, -0.02]
    lm[11] = [0.48, 0.45, -0.02]
    lm[12] = [0.48, 0.38, -0.02]
    # Ring extended
    lm[14] = [0.54, 0.57, -0.02]
    lm[15] = [0.55, 0.47, -0.02]
    lm[16] = [0.56, 0.40, -0.02]
    # Pinky extended
    lm[18] = [0.60, 0.62, -0.01]
    lm[19] = [0.62, 0.52, -0.01]
    lm[20] = [0.63, 0.45, -0.01]
    return lm


def _wave() -> np.ndarray:
    """Wave uses open palm as base (detection is temporal)."""
    return _open_palm()


# Map gesture names to template generators
GESTURE_TEMPLATES = {
    "point_left": _point_left,
    "point_right": _point_right,
    "point_up": _point_up,
    "point_down": _point_down,
    "open_palm": _open_palm,
    "thumbs_up": _thumbs_up,
    "thumbs_down": _thumbs_down,
    "pinch": _pinch,
    "wave": _wave,
}


# ═══════════════════════════════════════════════
# Augmentation functions
# ═══════════════════════════════════════════════

def add_noise(landmarks: np.ndarray, std: float = config.NOISE_STD) -> np.ndarray:
    """Add Gaussian noise to landmarks."""
    noise = np.random.normal(0, std, landmarks.shape).astype(np.float32)
    return landmarks + noise


def random_scale(landmarks: np.ndarray, scale_range: Tuple = config.SCALE_RANGE) -> np.ndarray:
    """Apply random uniform scaling around the wrist."""
    wrist = landmarks[0].copy()
    scale = np.random.uniform(*scale_range)
    scaled = (landmarks - wrist) * scale + wrist
    return scaled.astype(np.float32)


def random_translate(landmarks: np.ndarray, max_shift: float = 0.15) -> np.ndarray:
    """Apply random translation in x and y."""
    shift = np.random.uniform(-max_shift, max_shift, size=(1, 3)).astype(np.float32)
    shift[0, 2] = 0  # Don't shift z
    return landmarks + shift


def random_rotate_2d(landmarks: np.ndarray, max_degrees: float = config.ROTATION_RANGE) -> np.ndarray:
    """Apply random 2D rotation around the wrist in the xy-plane."""
    angle = np.radians(np.random.uniform(-max_degrees, max_degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    rotated = centered.copy()
    rotated[:, 0] = centered[:, 0] * cos_a - centered[:, 1] * sin_a
    rotated[:, 1] = centered[:, 0] * sin_a + centered[:, 1] * cos_a
    return (rotated + wrist).astype(np.float32)


def augment_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Apply full augmentation pipeline to landmarks."""
    lm = landmarks.copy()
    lm = add_noise(lm)
    lm = random_scale(lm)
    lm = random_translate(lm)
    lm = random_rotate_2d(lm)
    return lm


# ═══════════════════════════════════════════════
# Dataset generation
# ═══════════════════════════════════════════════

def generate_dataset(
    samples_per_gesture: int = config.SAMPLES_PER_GESTURE,
    output_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature dataset for all gesture classes.

    Parameters
    ----------
    samples_per_gesture : int
        Number of augmented samples per gesture class.
    output_dir : str, optional
        Directory to save the dataset. Defaults to config.DATA_DIR.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (features, labels) where features.shape = (N, feature_dim)
        and labels.shape = (N,) with integer class indices.
    """
    if output_dir is None:
        output_dir = config.DATA_DIR
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor()
    all_features = []
    all_labels = []

    print("=" * 60)
    print("Generating Synthetic Gesture Dataset")
    print("=" * 60)

    for gesture_name in config.GESTURE_CLASSES:
        template_fn = GESTURE_TEMPLATES[gesture_name]
        base_landmarks = template_fn()
        gesture_idx = config.GESTURE_TO_IDX[gesture_name]

        count = 0
        for _ in range(samples_per_gesture):
            # Augment landmarks
            augmented = augment_landmarks(base_landmarks)

            # Randomly assign handedness (feature extractor handles mirroring)
            handedness = np.random.choice(["Left", "Right"])

            try:
                features = extractor.extract(augmented, handedness=handedness)
                all_features.append(features)
                all_labels.append(gesture_idx)
                count += 1
            except Exception:
                continue

        print(f"  [{gesture_idx}] {gesture_name:15s} → {count:5d} samples generated")

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]

    # Save
    np.save(output_path / "features.npy", features)
    np.save(output_path / "labels.npy", labels)

    # Save metadata
    metadata = {
        "num_samples": len(features),
        "feature_dim": int(features.shape[1]),
        "num_classes": config.NUM_CLASSES,
        "classes": config.GESTURE_CLASSES,
        "samples_per_gesture": samples_per_gesture,
    }
    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTotal samples: {len(features)}")
    print(f"Feature dim:   {features.shape[1]}")
    print(f"Saved to:      {output_path}")
    print("=" * 60)

    return features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic gesture dataset")
    parser.add_argument(
        "--samples", type=int, default=config.SAMPLES_PER_GESTURE,
        help=f"Samples per gesture class (default: {config.SAMPLES_PER_GESTURE})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()
    generate_dataset(args.samples, args.output)
