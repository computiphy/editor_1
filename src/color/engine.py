"""
Color Grading Engine - AI Colorist
===================================
Implements the adjustment strategies defined in color_theory.md.
Supports 9 filter presets with per-channel HSL adjustments,
tone curves, split toning, vignette, and grain.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from src.core.enums import ColorMethod, ColorStyle


# ────────────────────────────────────────────────────────────────
# Preset Data Structures
# ────────────────────────────────────────────────────────────────

@dataclass
class ChannelHSL:
    hue: float = 0.0
    sat: float = 0.0
    lum: float = 0.0

@dataclass
class SplitTone:
    shadow_hue: Optional[float] = None
    shadow_saturation: float = 0.0
    highlight_hue: Optional[float] = None
    highlight_saturation: float = 0.0

@dataclass
class ToneCurve:
    shadows_lift: float = 0.0
    highlights_roll: float = 0.0
    midtone_gamma: float = 1.0

@dataclass
class StylePreset:
    name: str = "natural"
    # Global
    temperature_shift: float = 0.0
    tint_shift: float = 0.0
    exposure_offset: float = 0.0
    contrast: float = 1.0
    saturation_scale: float = 1.0
    vibrance_scale: float = 1.0
    # Tone
    tone_curve: ToneCurve = field(default_factory=ToneCurve)
    # Split Tone
    split_tone: SplitTone = field(default_factory=SplitTone)
    # Per-channel HSL
    per_channel: Dict[str, ChannelHSL] = field(default_factory=dict)
    # Effects
    vignette_strength: float = 0.0
    vignette_radius: float = 0.8
    grain_amount: float = 0.0
    grain_size: float = 1.0


# ────────────────────────────────────────────────────────────────
# Preset Registry (values from color_theory.md)
# ────────────────────────────────────────────────────────────────

def _ch(h=0, s=0, l=0):
    return ChannelHSL(hue=h, sat=s, lum=l)

PRESETS: Dict[str, StylePreset] = {
    "natural": StylePreset(
        name="natural",
        temperature_shift=3, contrast=1.05, vibrance_scale=1.05,
        per_channel={
            "red": _ch(), "orange": _ch(), "yellow": _ch(s=-5),
            "green": _ch(s=-10, l=-5), "blue": _ch(), "purple": _ch()
        }
    ),
    "cinematic": StylePreset(
        name="cinematic",
        temperature_shift=5, tint_shift=-2, exposure_offset=-0.1,
        contrast=1.15, saturation_scale=0.90, vibrance_scale=1.10,
        tone_curve=ToneCurve(shadows_lift=15, highlights_roll=-10, midtone_gamma=0.95),
        split_tone=SplitTone(shadow_hue=200, shadow_saturation=25,
                             highlight_hue=35, highlight_saturation=20),
        per_channel={
            "red": _ch(h=3, s=-5), "orange": _ch(h=-2, s=5, l=5),
            "yellow": _ch(h=-10, s=-15, l=-5), "green": _ch(h=15, s=-25, l=-10),
            "blue": _ch(h=-10, s=10, l=-10), "purple": _ch(h=5, s=-10, l=-5)
        },
        vignette_strength=0.3, vignette_radius=0.7,
        grain_amount=8, grain_size=1.5
    ),
    "pastel": StylePreset(
        name="pastel",
        temperature_shift=2, tint_shift=2, exposure_offset=0.3,
        contrast=0.85, saturation_scale=0.65, vibrance_scale=0.80,
        tone_curve=ToneCurve(shadows_lift=30, highlights_roll=-5, midtone_gamma=1.10),
        split_tone=SplitTone(shadow_hue=220, shadow_saturation=15,
                             highlight_hue=45, highlight_saturation=12),
        per_channel={
            "red": _ch(h=5, s=-20, l=10), "orange": _ch(s=-15, l=10),
            "yellow": _ch(h=-5, s=-20, l=15), "green": _ch(h=10, s=-30, l=10),
            "blue": _ch(h=5, s=-20, l=10), "purple": _ch(s=-15, l=10)
        },
        grain_amount=3
    ),
    "moody": StylePreset(
        name="moody",
        temperature_shift=-3, tint_shift=-3, exposure_offset=-0.4,
        contrast=1.25, saturation_scale=0.70, vibrance_scale=0.85,
        tone_curve=ToneCurve(shadows_lift=5, highlights_roll=-20, midtone_gamma=0.85),
        split_tone=SplitTone(shadow_hue=230, shadow_saturation=20,
                             highlight_hue=40, highlight_saturation=10),
        per_channel={
            "red": _ch(s=-10, l=-10), "orange": _ch(h=-3, s=-5, l=-5),
            "yellow": _ch(h=-8, s=-20, l=-15), "green": _ch(h=10, s=-30, l=-20),
            "blue": _ch(h=-5, s=5, l=-15), "purple": _ch(h=5, s=10, l=-10)
        },
        vignette_strength=0.5, vignette_radius=0.6,
        grain_amount=12, grain_size=2.0
    ),
    "golden_hour": StylePreset(
        name="golden_hour",
        temperature_shift=12, tint_shift=3, exposure_offset=0.1,
        contrast=1.10, saturation_scale=1.10, vibrance_scale=1.15,
        tone_curve=ToneCurve(shadows_lift=8, highlights_roll=-5, midtone_gamma=1.05),
        split_tone=SplitTone(shadow_hue=30, shadow_saturation=20,
                             highlight_hue=45, highlight_saturation=25),
        per_channel={
            "red": _ch(h=-3, s=10, l=5), "orange": _ch(h=-5, s=15, l=10),
            "yellow": _ch(h=-8, s=10, l=10), "green": _ch(h=-15, s=-20, l=-5),
            "blue": _ch(s=-15, l=-10), "purple": _ch(h=-10, s=-10, l=-5)
        },
        vignette_strength=0.2, vignette_radius=0.75,
        grain_amount=5, grain_size=1.2
    ),
    "film_kodak": StylePreset(
        name="film_kodak",
        temperature_shift=4, tint_shift=1, contrast=1.08,
        saturation_scale=0.92, vibrance_scale=1.05,
        tone_curve=ToneCurve(shadows_lift=12, highlights_roll=-8, midtone_gamma=1.02),
        split_tone=SplitTone(shadow_hue=160, shadow_saturation=8,
                             highlight_hue=40, highlight_saturation=15),
        per_channel={
            "red": _ch(h=2, s=-5, l=3), "orange": _ch(h=-3, s=8, l=5),
            "yellow": _ch(h=-5, s=-10, l=3), "green": _ch(h=8, s=-20, l=-8),
            "blue": _ch(h=-8, s=-5), "purple": _ch(h=3, s=-8, l=-3)
        },
        vignette_strength=0.15, vignette_radius=0.8,
        grain_amount=10, grain_size=1.8
    ),
    "film_fuji": StylePreset(
        name="film_fuji",
        temperature_shift=-2, tint_shift=2, exposure_offset=0.15,
        contrast=1.05, saturation_scale=0.88, vibrance_scale=1.08,
        tone_curve=ToneCurve(shadows_lift=18, highlights_roll=-5, midtone_gamma=1.05),
        split_tone=SplitTone(shadow_hue=185, shadow_saturation=15,
                             highlight_hue=50, highlight_saturation=10),
        per_channel={
            "red": _ch(h=2, s=-8, l=2), "orange": _ch(s=-3, l=5),
            "yellow": _ch(h=-3, s=-12, l=5), "green": _ch(h=5, s=-15, l=-3),
            "blue": _ch(h=-5, s=8, l=3), "purple": _ch(h=-5, s=-5)
        },
        vignette_strength=0.1, vignette_radius=0.85,
        grain_amount=6, grain_size=1.4
    ),
    "vibrant": StylePreset(
        name="vibrant",
        temperature_shift=3, exposure_offset=0.05,
        contrast=1.20, saturation_scale=1.30, vibrance_scale=1.25,
        tone_curve=ToneCurve(midtone_gamma=0.98),
        per_channel={
            "red": _ch(s=15), "orange": _ch(h=-3, s=10, l=5),
            "yellow": _ch(s=10, l=5), "green": _ch(h=-5, s=10, l=-5),
            "blue": _ch(s=15, l=-5), "purple": _ch(s=10)
        },
        vignette_strength=0.1
    ),
    "black_and_white": StylePreset(
        name="black_and_white",
        contrast=1.20, saturation_scale=0.0, vibrance_scale=0.0,
        tone_curve=ToneCurve(shadows_lift=5, highlights_roll=-5, midtone_gamma=0.95),
        per_channel={
            "red": _ch(l=10), "orange": _ch(l=15), "yellow": _ch(l=5),
            "green": _ch(l=-15), "blue": _ch(l=-20), "purple": _ch(l=-10)
        },
        vignette_strength=0.3, vignette_radius=0.7,
        grain_amount=15, grain_size=2.5
    ),
    "moody_forest": StylePreset(
        name="moody_forest",
        temperature_shift=-5, tint_shift=2, exposure_offset=-0.2,
        contrast=1.10, saturation_scale=0.85, vibrance_scale=1.0,
        tone_curve=ToneCurve(shadows_lift=5, highlights_roll=-15, midtone_gamma=0.90),
        split_tone=SplitTone(shadow_hue=210, shadow_saturation=10),
        per_channel={
            "red": _ch(), "orange": _ch(s=5, l=5),
            "yellow": _ch(h=-10, s=-20, l=-5), "green": _ch(h=10, s=-40, l=-15),
            "blue": _ch(h=-5, s=-10, l=-10), "purple": _ch(s=-20)
        },
        vignette_strength=0.4, vignette_radius=0.6,
        grain_amount=5, grain_size=1.2
    ),
    "golden_hour_portrait": StylePreset(
        name="golden_hour_portrait",
        temperature_shift=8, tint_shift=1, exposure_offset=0.1,
        contrast=1.0, saturation_scale=1.1, vibrance_scale=1.1,
        tone_curve=ToneCurve(shadows_lift=10, highlights_roll=-5, midtone_gamma=1.05),
        split_tone=SplitTone(shadow_hue=40, shadow_saturation=15, highlight_hue=50, highlight_saturation=10),
        per_channel={
            "red": _ch(h=5, s=5), "orange": _ch(s=10, l=5),
            "yellow": _ch(h=-5, s=15), "green": _ch(h=-10, s=5),
            "blue": _ch(h=-10, s=-15, l=5), "purple": _ch()
        },
        vignette_strength=0.2, vignette_radius=0.9
    ),
    "urban_cyberpunk": StylePreset(
        name="urban_cyberpunk",
        temperature_shift=-10, tint_shift=15, contrast=1.20,
        saturation_scale=1.2, vibrance_scale=1.3, exposure_offset=0.0,
        tone_curve=ToneCurve(shadows_lift=5, highlights_roll=0, midtone_gamma=0.95),
        split_tone=SplitTone(shadow_hue=280, shadow_saturation=30, highlight_hue=180, highlight_saturation=20),
        per_channel={
            "red": _ch(h=10, s=10), "orange": _ch(l=10),
            "yellow": _ch(h=-20, s=-10), "green": _ch(h=40, s=-20),
            "blue": _ch(h=-10, s=20, l=10), "purple": _ch(h=10, s=25, l=5)
        },
        vignette_strength=0.4, vignette_radius=0.6
    ),
    "vintage_painterly": StylePreset(
        name="vintage_painterly",
        temperature_shift=4, tint_shift=5, exposure_offset=-0.1,
        contrast=0.95, saturation_scale=0.85, vibrance_scale=0.9,
        tone_curve=ToneCurve(shadows_lift=25, highlights_roll=-20, midtone_gamma=1.0),
        split_tone=SplitTone(shadow_hue=210, shadow_saturation=8, highlight_hue=45, highlight_saturation=15),
        per_channel={
            "red": _ch(s=-10, l=-5), "orange": _ch(h=-2, s=-5),
            "yellow": _ch(h=-5, s=-10, l=-5), "green": _ch(h=-10, s=-30, l=-10),
            "blue": _ch(h=-5, s=-20, l=-10), "purple": _ch(s=-30)
        },
        vignette_strength=0.2, vignette_radius=0.5,
        grain_amount=15, grain_size=2.0
    ),
    "high_fashion": StylePreset(
        name="high_fashion",
        exposure_offset=0.2, contrast=1.15,
        saturation_scale=1.05, vibrance_scale=1.1,
        tone_curve=ToneCurve(shadows_lift=0, highlights_roll=-5, midtone_gamma=1.0),
        split_tone=SplitTone(shadow_hue=240, shadow_saturation=10),
        per_channel={
            "red": _ch(s=15), "orange": _ch(s=5, l=5),
            "yellow": _ch(), "green": _ch(h=20, s=10, l=-5),
            "blue": _ch(s=10), "purple": _ch(h=10, s=15)
        }
    ),
    "sepia_monochrome": StylePreset(
        name="sepia_monochrome",
        temperature_shift=10, contrast=1.10,
        saturation_scale=0.0, vibrance_scale=0.0,
        tone_curve=ToneCurve(shadows_lift=10, highlights_roll=-10, midtone_gamma=1.0),
        split_tone=SplitTone(shadow_hue=30, shadow_saturation=30, highlight_hue=45, highlight_saturation=20),
        per_channel={
            "red": _ch(l=10), "orange": _ch(l=15),
            "yellow": _ch(l=5), "green": _ch(l=-10),
            "blue": _ch(l=-20), "purple": _ch()
        },
        vignette_strength=0.3, vignette_radius=0.7,
        grain_amount=10, grain_size=1.2
    ),
    "vibrant_landscape": StylePreset(
        name="vibrant_landscape",
        contrast=1.15, saturation_scale=1.2, vibrance_scale=1.25,
        tone_curve=ToneCurve(highlights_roll=-15, midtone_gamma=1.0),
        per_channel={
            "red": _ch(s=10), "orange": _ch(s=10),
            "yellow": _ch(h=-5, s=15, l=5), "green": _ch(h=5, s=10, l=-5),
            "blue": _ch(h=-5, s=15, l=-10), "purple": _ch()
        },
        vignette_strength=0.1, vignette_radius=0.8
    ),
    "lavender_dream": StylePreset(
        name="lavender_dream",
        temperature_shift=-2, tint_shift=10, exposure_offset=0.25,
        contrast=0.90, saturation_scale=0.80, vibrance_scale=1.0,
        tone_curve=ToneCurve(shadows_lift=20, highlights_roll=-10, midtone_gamma=1.05),
        split_tone=SplitTone(shadow_hue=260, shadow_saturation=15, highlight_hue=330, highlight_saturation=10),
        per_channel={
            "red": _ch(h=10, s=-10, l=10), "orange": _ch(l=5),
            "yellow": _ch(h=-10, s=-20, l=10), "green": _ch(h=20, s=-40, l=10),
            "blue": _ch(h=10, s=-10, l=10), "purple": _ch(h=5, s=5, l=5)
        },
        vignette_strength=0.1, vignette_radius=0.8,
        grain_amount=5, grain_size=1.0
    ),
    "bleach_bypass": StylePreset(
        name="bleach_bypass",
        temperature_shift=-2, exposure_offset=-0.1,
        contrast=1.35, saturation_scale=0.40, vibrance_scale=0.50,
        tone_curve=ToneCurve(shadows_lift=0, highlights_roll=10, midtone_gamma=1.0),
        split_tone=SplitTone(shadow_hue=190, shadow_saturation=10, highlight_hue=30, highlight_saturation=10),
        per_channel={
            "red": _ch(s=10, l=-10), "orange": _ch(s=20),
            "yellow": _ch(s=-50), "green": _ch(s=-60, l=-20),
            "blue": _ch(s=-50, l=-10), "purple": _ch(s=-50)
        },
        vignette_strength=0.5, vignette_radius=0.6,
        grain_amount=20, grain_size=1.5
    ),
    "dark_academic": StylePreset(
        name="dark_academic",
        temperature_shift=2, exposure_offset=-0.3,
        contrast=1.05, saturation_scale=0.90, vibrance_scale=1.0,
        tone_curve=ToneCurve(shadows_lift=10, highlights_roll=-20, midtone_gamma=0.95),
        split_tone=SplitTone(shadow_hue=140, shadow_saturation=15, highlight_hue=35, highlight_saturation=10),
        per_channel={
            "red": _ch(s=-10, l=-5), "orange": _ch(h=-5, s=5),
            "yellow": _ch(h=-15, s=-20, l=-10), "green": _ch(h=10, s=-10, l=-10),
            "blue": _ch(h=-10, s=-40, l=-20), "purple": _ch(s=-40, l=-10)
        },
        vignette_strength=0.3, vignette_radius=0.7,
        grain_amount=5, grain_size=1.0
    ),}


# ────────────────────────────────────────────────────────────────
# Color Grading Engine
# ────────────────────────────────────────────────────────────────

# OpenCV HSV hue ranges for color channels (H is 0-180 in OpenCV)
HSL_RANGES = {
    "red":    [(0, 10), (170, 180)],  # Red wraps around
    "orange": [(10, 25)],
    "yellow": [(25, 35)],
    "green":  [(35, 85)],
    "blue":   [(85, 130)],
    "purple": [(130, 170)],
}


class ColorGradingEngine:
    """
    AI Colorist Engine.
    Applies filter presets with per-channel HSL, tone curves,
    split toning, vignette, and grain.
    """

    def __init__(self, method: ColorMethod = ColorMethod.LAB_STATISTICAL,
                 style: str = "natural", strength: float = 1.0):
        self.method = method
        self.strength = np.clip(strength, 0.0, 1.5)
        self.preset = PRESETS.get(style, PRESETS["natural"])

    # ── Public API ──────────────────────────────────────────────

    def apply_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Reference-based Reinhard transfer (legacy API)."""
        return self._reinhard_transfer(source, target)

    def apply_style(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full style preset pipeline.
        This is the main entry point for the AI colorist.
        """
        img = image.copy().astype(np.float32)

        # 1. White Balance (Temperature / Tint)
        img = self._adjust_temperature(img, self.preset.temperature_shift * self.strength)
        img = self._adjust_tint(img, self.preset.tint_shift * self.strength)

        # 2. Exposure
        img = self._adjust_exposure(img, self.preset.exposure_offset * self.strength)

        # 3. Tone Curve (shadows lift, highlights roll, gamma)
        img = self._apply_tone_curve(img, self.preset.tone_curve)

        # 4. Contrast (S-curve in L channel)
        img = self._adjust_contrast(img, self.preset.contrast)

        # 5. Per-Channel HSL Adjustments
        img = self._apply_per_channel_hsl(img, self.preset.per_channel)

        # 6. Split Toning
        img = self._apply_split_tone(img, self.preset.split_tone)

        # 7. Saturation & Vibrance
        img = self._adjust_saturation(img, self.preset.saturation_scale)
        img = self._adjust_vibrance(img, self.preset.vibrance_scale)

        # 8. Vignette
        if self.preset.vignette_strength > 0:
            img = self._apply_vignette(img, self.preset.vignette_strength,
                                       self.preset.vignette_radius)

        # 9. Grain
        if self.preset.grain_amount > 0:
            img = self._apply_grain(img, self.preset.grain_amount,
                                    self.preset.grain_size)

        return np.clip(img, 0, 255).astype(np.uint8)

    # ── Temperature & Tint ──────────────────────────────────────

    def _adjust_temperature(self, img: np.ndarray, shift: float) -> np.ndarray:
        """Shift temperature. Positive = warm (add R, reduce B). Negative = cool."""
        if abs(shift) < 0.5:
            return img
        scale = shift * 2.5  # Map Kelvin-like shift to pixel offset
        img[:, :, 0] = img[:, :, 0] + scale        # Red
        img[:, :, 2] = img[:, :, 2] - scale * 0.7  # Blue (inverse)
        return img

    def _adjust_tint(self, img: np.ndarray, shift: float) -> np.ndarray:
        """Shift tint. Positive = magenta. Negative = green."""
        if abs(shift) < 0.5:
            return img
        scale = shift * 2.0
        img[:, :, 1] = img[:, :, 1] - scale  # Green channel
        return img

    # ── Exposure ────────────────────────────────────────────────

    def _adjust_exposure(self, img: np.ndarray, offset: float) -> np.ndarray:
        """Adjust exposure in stops. +1 = double brightness."""
        if abs(offset) < 0.01:
            return img
        factor = 2.0 ** offset
        return img * factor

    # ── Tone Curve ──────────────────────────────────────────────

    def _apply_tone_curve(self, img: np.ndarray, tc: ToneCurve) -> np.ndarray:
        """Apply shadow lift, highlight roll, and midtone gamma."""
        # Build LUT
        lut = np.arange(256, dtype=np.float32)

        # Gamma (midtones)
        if abs(tc.midtone_gamma - 1.0) > 0.01:
            lut = 255.0 * (lut / 255.0) ** (1.0 / tc.midtone_gamma)

        # Shadow lift (remap 0 → shadows_lift)
        if tc.shadows_lift > 0:
            lift = tc.shadows_lift * self.strength
            lut = lift + lut * (255.0 - lift) / 255.0

        # Highlight roll (remap 255 → 255 + highlights_roll)
        if tc.highlights_roll < 0:
            ceiling = 255.0 + tc.highlights_roll * self.strength
            lut = lut * ceiling / 255.0

        lut = np.clip(lut, 0, 255).astype(np.uint8)

        # Apply LUT to each channel
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        for c in range(3):
            img_u8[:, :, c] = cv2.LUT(img_u8[:, :, c], lut)

        return img_u8.astype(np.float32)

    # ── Contrast ────────────────────────────────────────────────

    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast via LAB L-channel scaling around midpoint."""
        if abs(factor - 1.0) < 0.01:
            return img
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2Lab).astype(np.float32)
        l_ch = lab[:, :, 0]

        mid = 128.0
        effective = 1.0 + (factor - 1.0) * self.strength
        l_ch = mid + (l_ch - mid) * effective

        lab[:, :, 0] = np.clip(l_ch, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
        return result.astype(np.float32)

    # ── Per-Channel HSL ─────────────────────────────────────────

    def _apply_per_channel_hsl(self, img: np.ndarray,
                                channels: Dict[str, ChannelHSL]) -> np.ndarray:
        """Apply hue/saturation/luminance shifts per color range."""
        if not channels:
            return img

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        for color_name, adj in channels.items():
            if abs(adj.hue) < 0.5 and abs(adj.sat) < 0.5 and abs(adj.lum) < 0.5:
                continue

            ranges = HSL_RANGES.get(color_name, [])
            mask = np.zeros(h_ch.shape, dtype=bool)
            for (lo, hi) in ranges:
                mask |= (h_ch >= lo) & (h_ch <= hi)

            # Also require some saturation (exclude greys)
            mask &= (s_ch > 20)

            if not mask.any():
                continue

            # Apply scaled adjustments
            h_ch[mask] += adj.hue * self.strength * 0.5  # OpenCV H is 0-180
            s_ch[mask] += adj.sat * self.strength * 1.28  # Scale % to 0-255
            v_ch[mask] += adj.lum * self.strength * 1.28

        # Wrap hue
        h_ch = h_ch % 180

        hsv[:, :, 0] = np.clip(h_ch, 0, 179)
        hsv[:, :, 1] = np.clip(s_ch, 0, 255)
        hsv[:, :, 2] = np.clip(v_ch, 0, 255)

        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32)

    # ── Split Toning ────────────────────────────────────────────

    def _apply_split_tone(self, img: np.ndarray, st: SplitTone) -> np.ndarray:
        """Colorize shadows and highlights independently."""
        if st.shadow_hue is None and st.highlight_hue is None:
            return img

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2Lab).astype(np.float32)
        l_ch = lab[:, :, 0]

        # Normalize luminance to 0-1 for blending masks
        l_norm = l_ch / 255.0

        # Shadow mask: strongest at L=0, fades by L=128
        shadow_mask = np.clip(1.0 - l_norm * 2, 0, 1)
        # Highlight mask: strongest at L=255, fades by L=128
        highlight_mask = np.clip(l_norm * 2 - 1, 0, 1)

        if st.shadow_hue is not None and st.shadow_saturation > 0:
            # Convert hue (0-360) to LAB a,b offset
            rad = np.radians(st.shadow_hue)
            a_offset = np.cos(rad) * st.shadow_saturation * self.strength * 0.5
            b_offset = np.sin(rad) * st.shadow_saturation * self.strength * 0.5
            lab[:, :, 1] += shadow_mask * a_offset
            lab[:, :, 2] += shadow_mask * b_offset

        if st.highlight_hue is not None and st.highlight_saturation > 0:
            rad = np.radians(st.highlight_hue)
            a_offset = np.cos(rad) * st.highlight_saturation * self.strength * 0.5
            b_offset = np.sin(rad) * st.highlight_saturation * self.strength * 0.5
            lab[:, :, 1] += highlight_mask * a_offset
            lab[:, :, 2] += highlight_mask * b_offset

        lab = np.clip(lab, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
        return result.astype(np.float32)

    # ── Saturation & Vibrance ───────────────────────────────────

    def _adjust_saturation(self, img: np.ndarray, scale: float) -> np.ndarray:
        """Scale saturation uniformly."""
        effective = 1.0 + (scale - 1.0) * self.strength
        if abs(effective - 1.0) < 0.01:
            return img

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= effective
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32)

    def _adjust_vibrance(self, img: np.ndarray, scale: float) -> np.ndarray:
        """Boost under-saturated colors more than saturated ones."""
        effective = 1.0 + (scale - 1.0) * self.strength
        if abs(effective - 1.0) < 0.01:
            return img

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Weight: low-saturation pixels get a stronger boost
        s_norm = hsv[:, :, 1] / 255.0
        weight = 1.0 - s_norm  # Higher weight for desaturated pixels
        boost = 1.0 + (effective - 1.0) * weight

        hsv[:, :, 1] *= boost
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32)

    # ── Vignette ────────────────────────────────────────────────

    def _apply_vignette(self, img: np.ndarray, strength: float,
                        radius: float) -> np.ndarray:
        """Radial darkening from center."""
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2

        # Distance from center, normalized
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist_norm = dist / max_dist

        # Vignette mask: 1 at center, drops off by radius
        mask = 1.0 - np.clip((dist_norm - radius) / (1.0 - radius + 1e-5), 0, 1)
        mask = mask ** 2  # Smooth falloff
        vignette = 1.0 - strength * self.strength * (1.0 - mask)

        return img * vignette[:, :, np.newaxis]

    # ── Grain ───────────────────────────────────────────────────

    def _apply_grain(self, img: np.ndarray, amount: float,
                     size: float) -> np.ndarray:
        """Add photographic grain (Gaussian noise)."""
        h, w = img.shape[:2]
        # Generate at reduced resolution for larger grain
        gh, gw = int(h / size), int(w / size)
        noise = np.random.normal(0, amount * self.strength, (gh, gw))
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        noise = np.stack([noise] * 3, axis=-1)
        return img + noise

    # ── Reinhard Transfer (Legacy) ──────────────────────────────

    def _reinhard_transfer(self, source: np.ndarray,
                           target: np.ndarray) -> np.ndarray:
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2Lab).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2Lab).astype(np.float32)

        s_mean, s_std = self._get_stats(source_lab)
        t_mean, t_std = self._get_stats(target_lab)

        res_lab = (source_lab - s_mean) * (t_std / s_std) + t_mean
        res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(res_lab, cv2.COLOR_Lab2RGB)

    def _get_stats(self, img_lab: np.ndarray):
        (l, a, b) = cv2.split(img_lab)
        means = np.array([l.mean(), a.mean(), b.mean()])
        stds = np.array([max(l.std(), 1e-5), max(a.std(), 1e-5), max(b.std(), 1e-5)])
        return means, stds
