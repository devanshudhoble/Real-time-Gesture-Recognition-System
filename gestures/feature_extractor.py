"""
Feature Extractor Module
========================
Converts raw 21-landmark hand data from MediaPipe into a rich, normalized
feature vector suitable for gesture classification.

Feature Engineering Strategy:
    Instead of feeding raw (x, y, z) coordinates (which are camera-dependent),
    we compute geometric features that are invariant to:
      - Translation (re-center on wrist)
      - Scale (normalize by palm size)
      - Camera distance (use ratios and angles)

    Feature groups:
      1. Relative landmark positions (wrist-centered, palm-normalized)
      2. Finger extension ratios (tip-to-MCP vs PIP-to-MCP distances)
      3. Inter-finger tip distances
      4. Finger curl angles
      5. Palm orientation (roll, pitch derived from key landmarks)
      6. Thumb-index distance (for pinch detection)

    Total features: 78 per hand (configurable via config.INPUT_FEATURE_DIM)
"""

import numpy as np
from typing import Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config


class FeatureExtractor:
    """
    Extracts a normalized, geometry-invariant feature vector from hand landmarks.

    The feature vector is designed to be robust to:
      - Varying hand sizes and distances from camera
      - Different lighting conditions (landmarks, not pixels)
      - Left vs right hand (mirrored if needed)
    """

    def __init__(self):
        self.num_landmarks = config.NUM_LANDMARKS

    def extract(self, landmarks: np.ndarray, handedness: str = "Right") -> np.ndarray:
        """
        Extract feature vector from hand landmarks.

        Parameters
        ----------
        landmarks : np.ndarray
            Shape (21, 3) — normalized (x, y, z) from MediaPipe.
        handedness : str
            "Left" or "Right". Left hands are mirrored to Right for consistency.

        Returns
        -------
        np.ndarray
            1-D feature vector of length config.INPUT_FEATURE_DIM.
        """
        lm = landmarks.copy()

        # Mirror left hand to right for consistency
        if handedness == "Left":
            lm[:, 0] = 1.0 - lm[:, 0]

        # ── 1. Re-center on wrist ──
        wrist = lm[config.WRIST].copy()
        centered = lm - wrist

        # ── 2. Normalize by palm size ──
        palm_size = np.linalg.norm(
            lm[config.MIDDLE_MCP] - lm[config.WRIST]
        )
        if palm_size < 1e-6:
            palm_size = 1e-6
        normalized = centered / palm_size

        # ── Build feature groups ──
        features = []

        # Group 1: Relative normalized positions (21 × 3 = 63 features, skip wrist = 60)
        rel_positions = normalized[1:].flatten()  # skip wrist (always 0,0,0)
        features.append(rel_positions)

        # Group 2: Finger extension ratios (5 features)
        extension_ratios = self._finger_extension_ratios(lm)
        features.append(extension_ratios)

        # Group 3: Inter-finger tip distances (C(5,2) = 10 features)
        tip_distances = self._inter_finger_distances(normalized)
        features.append(tip_distances)

        # Group 4: Finger curl angles — not used, replaced by extension ratios
        # (angles are less stable with noisy landmarks)

        # Group 5: Thumb-index specific distance (1 feature — key for pinch)
        thumb_index_dist = np.linalg.norm(
            normalized[config.THUMB_TIP] - normalized[config.INDEX_TIP]
        )
        features.append(np.array([thumb_index_dist]))

        # Group 6: Palm orientation angles (2 features)
        palm_angles = self._palm_orientation(normalized)
        features.append(palm_angles)

        feature_vector = np.concatenate(features).astype(np.float32)
        return feature_vector

    @staticmethod
    def get_feature_dim() -> int:
        """Return the expected feature dimension."""
        # 60 (rel positions) + 5 (extension) + 10 (tip dists) + 1 (thumb-idx) + 2 (palm orient) = 78
        return 78

    # ──────────────────── private helpers ────────────────────

    def _finger_extension_ratios(self, lm: np.ndarray) -> np.ndarray:
        """
        Compute extension ratio for each finger.

        Ratio = dist(tip, MCP) / dist(PIP, MCP)
        High ratio → finger extended, Low ratio → finger curled.
        """
        tips = [config.THUMB_TIP, config.INDEX_TIP, config.MIDDLE_TIP,
                config.RING_TIP, config.PINKY_TIP]
        mcps = [config.THUMB_MCP, config.INDEX_MCP, config.MIDDLE_MCP,
                config.RING_MCP, config.PINKY_MCP]
        pips = [config.THUMB_IP, config.INDEX_PIP, config.MIDDLE_PIP,
                config.RING_PIP, config.PINKY_PIP]

        ratios = []
        for tip, mcp, pip_ in zip(tips, mcps, pips):
            tip_mcp = np.linalg.norm(lm[tip] - lm[mcp])
            pip_mcp = np.linalg.norm(lm[pip_] - lm[mcp])
            ratio = tip_mcp / (pip_mcp + 1e-6)
            ratios.append(ratio)
        return np.array(ratios, dtype=np.float32)

    def _inter_finger_distances(self, normalized: np.ndarray) -> np.ndarray:
        """
        Pairwise distances between all finger tips.
        Returns 10 distances (C(5,2) combinations).
        """
        tips = [config.THUMB_TIP, config.INDEX_TIP, config.MIDDLE_TIP,
                config.RING_TIP, config.PINKY_TIP]
        dists = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(normalized[tips[i]] - normalized[tips[j]])
                dists.append(d)
        return np.array(dists, dtype=np.float32)

    def _palm_orientation(self, normalized: np.ndarray) -> np.ndarray:
        """
        Estimate palm orientation using vectors between key landmarks.
        Returns 2 angles: pitch-like and roll-like.
        """
        # Vector from wrist to middle MCP (normalized space, wrist is origin)
        palm_vec = normalized[config.MIDDLE_MCP]
        norm = np.linalg.norm(palm_vec[:2])
        if norm < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Angle of palm direction in xy plane
        angle_xy = np.arctan2(palm_vec[1], palm_vec[0])

        # Angle of palm tilt (z component)
        angle_z = np.arctan2(palm_vec[2], norm)

        return np.array([angle_xy, angle_z], dtype=np.float32)
