"""
Semantic Segmenter
==================
Produces binary masks for key regions in wedding photos:
skin, sky, vegetation, white dress, dark suit.

Uses OpenCV HSV color-space analysis (the ranges from color_theory.md §3.2)
combined with luminance heuristics. Masks are feathered with Gaussian blur
for smooth blending during semantic color grading.

Segmentation Classes (from color_theory.md §6.1):
  0 - skin          (HSV color detection)
  1 - sky           (HSV color + top-of-frame heuristic)
  2 - vegetation    (HSV color detection)
  5 - dress_white   (high luminance + low saturation)
  6 - suit_dark     (low luminance + low saturation)
"""

import cv2
import numpy as np
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class SegmentationResult:
    """Container for all detected region masks."""
    skin: np.ndarray        # (H, W) float32, 0.0–1.0
    sky: np.ndarray         # (H, W) float32, 0.0–1.0
    vegetation: np.ndarray  # (H, W) float32, 0.0–1.0
    dress_white: np.ndarray # (H, W) float32, 0.0–1.0
    suit_dark: np.ndarray   # (H, W) float32, 0.0–1.0

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "skin": self.skin,
            "sky": self.sky,
            "vegetation": self.vegetation,
            "dress_white": self.dress_white,
            "suit_dark": self.suit_dark,
        }


class SemanticSegmenter:
    """
    HSV + Luminance based semantic segmenter for wedding photos.

    Produces soft float masks (0.0–1.0) with Gaussian-feathered edges.
    These masks are used by the ColorGradingEngine to apply per-region
    adjustment overrides on top of the global preset.
    """

    def __init__(self, feather_sigma: int = 7):
        """
        Args:
            feather_sigma: Gaussian blur kernel size for mask edge feathering.
                           Must be odd. Larger = softer transitions.
        """
        self.feather_sigma = feather_sigma if feather_sigma % 2 == 1 else feather_sigma + 1

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Segment an RGB image into semantic regions.

        Args:
            image: RGB uint8 numpy array (H, W, 3).

        Returns:
            SegmentationResult with soft float masks for each region.
        """
        h, w = image.shape[:2]

        # Convert to HSV (OpenCV scale: H=0-180, S=0-255, V=0-255)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_ch = hsv[:, :, 0].astype(np.float32)
        s_ch = hsv[:, :, 1].astype(np.float32)
        v_ch = hsv[:, :, 2].astype(np.float32)

        # Convert to LAB for luminance
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l_ch = lab[:, :, 0].astype(np.float32)

        # ── 1. Skin Detection ──────────────────────────────────────
        # color_theory.md §3.2: H: 5–25, S: 40–170, V: 80–255
        skin_mask = (
            (h_ch >= 5) & (h_ch <= 25) &
            (s_ch >= 40) & (s_ch <= 170) &
            (v_ch >= 80)
        ).astype(np.float32)
        # Morphological cleanup: remove noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # ── 2. Sky Detection ──────────────────────────────────────
        # color_theory.md §3.2: H: 95–125, S: 50–255, V: 80–255
        sky_color = (
            (h_ch >= 95) & (h_ch <= 125) &
            (s_ch >= 50) &
            (v_ch >= 80)
        ).astype(np.float32)
        # Position heuristic: sky is more likely in top 60% of frame
        position_weight = np.ones((h, w), dtype=np.float32)
        gradient = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(-1, 1)
        position_weight = np.broadcast_to(gradient, (h, w)).copy()
        sky_mask = sky_color * position_weight
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

        # ── 3. Vegetation Detection ──────────────────────────────
        # color_theory.md §3.2: H: 30–85, S: 40–255, V: 30–200
        veg_mask = (
            (h_ch >= 30) & (h_ch <= 85) &
            (s_ch >= 40) &
            (v_ch >= 30) & (v_ch <= 200)
        ).astype(np.float32)
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)

        # ── 4. White Dress Detection ──────────────────────────────
        # color_theory.md §3.2: H: 0–30, S: 0–40, V: 200–255
        # High luminance + very low saturation = white fabric
        dress_mask = (
            (s_ch <= 40) &
            (v_ch >= 200) &
            (l_ch >= 200)
        ).astype(np.float32)
        # Remove small white specular highlights (keep large regions only)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dress_mask = cv2.morphologyEx(dress_mask, cv2.MORPH_OPEN, kernel_large)
        dress_mask = cv2.morphologyEx(dress_mask, cv2.MORPH_CLOSE, kernel_large)

        # ── 5. Dark Suit Detection ──────────────────────────────
        # Low luminance + low saturation = dark formal clothing
        suit_mask = (
            (l_ch <= 50) &
            (s_ch <= 80) &
            (v_ch <= 80)
        ).astype(np.float32)
        suit_mask = cv2.morphologyEx(suit_mask, cv2.MORPH_OPEN, kernel)
        suit_mask = cv2.morphologyEx(suit_mask, cv2.MORPH_CLOSE, kernel)

        # ── Priority Resolution ──────────────────────────────────
        # Skin has HIGHEST priority: remove skin pixels from other masks
        skin_binary = (skin_mask > 0.5).astype(np.float32)
        sky_mask = sky_mask * (1.0 - skin_binary)
        veg_mask = veg_mask * (1.0 - skin_binary)
        dress_mask = dress_mask * (1.0 - skin_binary)
        suit_mask = suit_mask * (1.0 - skin_binary)

        # Dress and suit also take priority over vegetation
        dress_binary = (dress_mask > 0.5).astype(np.float32)
        suit_binary = (suit_mask > 0.5).astype(np.float32)
        veg_mask = veg_mask * (1.0 - dress_binary) * (1.0 - suit_binary)

        # ── Feather all mask edges ───────────────────────────────
        # color_theory.md §6.3 step 5: Gaussian blur for smooth transitions
        sigma = self.feather_sigma
        skin_mask = cv2.GaussianBlur(skin_mask, (sigma, sigma), 0)
        sky_mask = cv2.GaussianBlur(sky_mask, (sigma, sigma), 0)
        veg_mask = cv2.GaussianBlur(veg_mask, (sigma, sigma), 0)
        dress_mask = cv2.GaussianBlur(dress_mask, (sigma, sigma), 0)
        suit_mask = cv2.GaussianBlur(suit_mask, (sigma, sigma), 0)

        return SegmentationResult(
            skin=skin_mask,
            sky=sky_mask,
            vegetation=veg_mask,
            dress_white=dress_mask,
            suit_dark=suit_mask,
        )
