"""
Central configuration for the gesture recognition system.

All hyperparameters, gesture labels, MediaPipe settings, and paths are
defined here so every module draws from a single source of truth.
"""

import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Gesture classes
# ──────────────────────────────────────────────
GESTURE_CLASSES = [
    "point_left",    # 0 — Index finger pointing left
    "point_right",   # 1 — Index finger pointing right
    "point_up",      # 2 — Index finger pointing up
    "point_down",    # 3 — Index finger pointing down
    "open_palm",     # 4 — All fingers extended (stop gesture)
    "thumbs_up",     # 5 — Thumb up, other fingers curled
    "thumbs_down",   # 6 — Thumb down, other fingers curled
    "pinch",         # 7 — Thumb and index finger close together
    "wave",          # 8 — Detected via temporal oscillation pattern
]

NUM_CLASSES = len(GESTURE_CLASSES)

GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURE_CLASSES)}
IDX_TO_GESTURE = {i: g for i, g in enumerate(GESTURE_CLASSES)}

# ──────────────────────────────────────────────
# MediaPipe hand detector settings
# ──────────────────────────────────────────────
MEDIAPIPE_MAX_HANDS = 2
MEDIAPIPE_MODEL_COMPLEXITY = 1          # 0 = lite, 1 = full
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Number of landmarks per hand (MediaPipe standard)
NUM_LANDMARKS = 21

# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
# Landmark indices (MediaPipe hand model)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_DIPS = [THUMB_MCP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]

# ──────────────────────────────────────────────
# Classifier model hyper-parameters
# ──────────────────────────────────────────────
INPUT_FEATURE_DIM = 78     # Set after feature extractor is finalized
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DROPOUT_RATE = 0.3

# ──────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────
TRAIN_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 10            # ReduceLROnPlateau patience
EARLY_STOP_PATIENCE = 15
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ──────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────
SAMPLES_PER_GESTURE = 800   # Synthetic samples per class
NOISE_STD = 0.02            # Gaussian noise std for augmentation
ROTATION_RANGE = 15         # Degrees of random rotation augmentation
SCALE_RANGE = (0.8, 1.2)    # Random scale augmentation range

# ──────────────────────────────────────────────
# Post-processing / temporal smoothing
# ──────────────────────────────────────────────
EMA_ALPHA = 0.6             # Exponential moving average smoothing factor
CONFIDENCE_THRESHOLD = 0.55 # Minimum confidence to report a gesture
HYSTERESIS_FRAMES = 3       # Consecutive frames needed to switch gesture
WAVE_WINDOW_SIZE = 15       # Number of frames to detect wave pattern
WAVE_MIN_OSCILLATIONS = 2   # Minimum direction changes for wave

# ──────────────────────────────────────────────
# Demo / inference
# ──────────────────────────────────────────────
# On macOS, Continuity Camera often claims index 0 while the built-in
# MacBook camera appears at index 1. Allow an env override for portability.
CAMERA_INDEX = int(
    os.getenv("GESTURE_CAMERA_INDEX", "1" if sys.platform == "darwin" else "0")
)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ──────────────────────────────────────────────
# Model file names
# ──────────────────────────────────────────────
BEST_MODEL_PATH = MODELS_DIR / "gesture_classifier_best.pth"
ONNX_MODEL_PATH = MODELS_DIR / "gesture_classifier.onnx"
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = MODELS_DIR / "classification_report.txt"
