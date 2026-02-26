"""
SOTA Color Engine V2
=====================
State-of-the-art color grading engine built on:
  - 32-bit float processing (no uint8 roundtrips)
  - Oklab perceptually uniform color space
  - Cubic spline tone curves
  - Subtractive saturation (filmic density emulation)

This module works alongside the existing engine.py. The legacy engine
is preserved for backward compatibility. V2 shares the same PRESETS
dictionary and StylePreset dataclass.

Processing Pipeline (in order):
  1. Input normalization (uint8 → float32 [0–1])
  2. sRGB → Linear (gamma decode)
  3. White Balance (Temperature/Tint in linear space)
  4. Exposure (in linear space — mathematically correct)
  5. Linear → Oklab
  6. Tone Curve (cubic spline, applied to Oklab L channel)
  7. Contrast (Oklab L channel S-curve)
  8. Per-Channel Hue/Saturation/Lightness (Oklab-based)
  9. Split Toning (Oklab a,b channels)
  10. Subtractive Saturation (filmic density)
  11. Vibrance (chroma-aware boost in Oklab)
  12. Oklab → sRGB
  13. Vignette
  14. Grain
  15. Output (float32 → uint8)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

from src.color.engine import PRESETS, StylePreset, ToneCurve, SplitTone, ChannelHSL, HSL_RANGES
from src.color.oklab import (
    srgb_to_linear, linear_to_srgb,
    srgb_to_oklab, oklab_to_srgb,
    oklab_chroma, oklab_hue,
)


# ────────────────────────────────────────────────────────────────
# Cubic Spline Tone Curve
# ────────────────────────────────────────────────────────────────

@dataclass
class SplineToneCurve:
    """
    High-precision tone curve using cubic spline interpolation.
    Replaces the legacy gamma/lift/ceiling formula.

    Nodes are (input, output) pairs in [0.0, 1.0] range.
    Example: [(0, 0), (0.25, 0.20), (0.75, 0.80), (1.0, 1.0)]
    """
    nodes: List[Tuple[float, float]]

    def generate_lut(self, size: int = 4096) -> np.ndarray:
        """
        Generate a smooth 1D LUT from the spline.

        Args:
            size: Number of entries. 4096 for float32 precision.

        Returns:
            float32 array of length `size`, clamped to [0, 1].
        """
        xs = np.array([n[0] for n in self.nodes], dtype=np.float64)
        ys = np.array([n[1] for n in self.nodes], dtype=np.float64)

        if len(xs) < 2:
            return np.linspace(0.0, 1.0, size, dtype=np.float32)

        # Use natural cubic spline (smooth at endpoints)
        cs = CubicSpline(xs, ys, bc_type='natural')
        t = np.linspace(0.0, 1.0, size, dtype=np.float64)
        lut = cs(t)
        return np.clip(lut, 0.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Subtractive Saturation
# ────────────────────────────────────────────────────────────────

def subtractive_saturate(oklab: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Apply subtractive (filmic) saturation in Oklab space.

    Unlike additive saturation (which makes colors brighter),
    subtractive saturation makes highly saturated colors DARKER
    and DENSER, emulating how real film dyes absorb light.

    Args:
        oklab: float32 array (H, W, 3) in Oklab space.
        factor: Saturation multiplier. >1.0 = boost, <1.0 = reduce.

    Returns:
        float32 array (H, W, 3) in Oklab space with modified saturation.
    """
    result = oklab.copy()
    L = result[:, :, 0]
    a = result[:, :, 1]
    b = result[:, :, 2]

    # Current chroma
    chroma = np.sqrt(a ** 2 + b ** 2)

    # Scale chroma
    new_chroma = chroma * factor
    delta_chroma = new_chroma - chroma

    # Subtractive: when chroma increases, L decreases proportionally
    # The density coefficient controls how much darkening occurs per unit of chroma boost.
    # Real Kodak Portra 400 has approximately 0.15 density per unit chroma.
    density_coeff = 0.15
    L_penalty = delta_chroma * density_coeff
    result[:, :, 0] = np.clip(L - L_penalty, 0.0, 1.0)

    # Scale a and b channels to achieve new chroma (preserving hue angle)
    scale = np.where(chroma > 1e-6, new_chroma / chroma, 1.0)
    result[:, :, 1] = a * scale
    result[:, :, 2] = b * scale

    return result


# ────────────────────────────────────────────────────────────────
# P2: CLAHE in Oklab
# ────────────────────────────────────────────────────────────────

def apply_clahe_oklab(image: np.ndarray, clip_limit: float = 2.0,
                      grid_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization to the Oklab
    L channel. This dramatically improves flat, muddy, or poorly lit images
    without over-saturating colors (unlike CLAHE in RGB or even CIELAB).

    Args:
        image: float32 array (H, W, 3) in sRGB [0-1].
        clip_limit: CLAHE clip limit (higher = more contrast).
        grid_size: CLAHE tile grid size.

    Returns:
        float32 array (H, W, 3) in sRGB [0-1] with enhanced contrast.
    """
    import cv2

    # Convert to Oklab
    oklab = srgb_to_oklab(image)

    # Extract L channel, scale to [0, 255] for OpenCV CLAHE (which needs uint8)
    L = oklab[:, :, 0]
    L_u8 = np.clip(L * 255, 0, 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    L_enhanced = clahe.apply(L_u8)

    # Map back to float32 [0, 1]
    oklab[:, :, 0] = L_enhanced.astype(np.float32) / 255.0

    # Convert back to sRGB
    return np.clip(oklab_to_srgb(oklab), 0.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# P2: Frequency Separation for Skin Grading
# ────────────────────────────────────────────────────────────────

def frequency_separate(image: np.ndarray,
                       blur_radius: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split an image into low-frequency (color/tone) and high-frequency
    (texture/detail) components.

    The key insight: modify only the low-frequency map for color correction
    on skin, then recombine with the untouched high-frequency. This fixes
    skin color casts while preserving 100% of pore/wrinkle texture,
    preventing the "plastic AI skin" look.

    Args:
        image: float32 array (H, W, 3) in [0-1].
        blur_radius: Gaussian blur kernel size (must be odd).

    Returns:
        (low_freq, high_freq) tuple of float32 arrays.
        low + high = original (within floating point precision).
    """
    import cv2

    # Ensure blur_radius is odd
    ksize = blur_radius * 2 + 1

    # Low frequency = Gaussian blur (smooth color/tone information)
    low = cv2.GaussianBlur(image, (ksize, ksize), 0).astype(np.float32)

    # High frequency = Original minus Low (texture, pores, wrinkles)
    high = (image - low).astype(np.float32)

    return low, high


def frequency_merge(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Recombine low-frequency and high-frequency components.

    Args:
        low: float32 low-frequency (color/tone) component.
        high: float32 high-frequency (texture) component.

    Returns:
        float32 reconstructed image.
    """
    return np.clip(low + high, 0.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# P3: Halation (Red-Channel Scattering)
# ────────────────────────────────────────────────────────────────

def apply_halation(image: np.ndarray, intensity: float = 0.3,
                   radius: int = 15, threshold: float = 0.7) -> np.ndarray:
    """
    Emulate film halation — the red glow around bright highlights caused
    by light scattering through the film base and reflecting off the
    anti-halation backing layer.

    Instead of generic bloom (which brightens all channels equally),
    this isolates high-luminance pixels, blurs ONLY the red channel,
    and composites it using an optical screen blend.

    Args:
        image: float32 array (H, W, 3) in sRGB [0-1].
        intensity: Strength of the halation effect (0 = off).
        radius: Blur radius for the red scatter.
        threshold: Luminance threshold for highlight detection.

    Returns:
        float32 array (H, W, 3) in sRGB [0-1].
    """
    if intensity <= 0:
        return image

    import cv2

    # 1. Compute luminance
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

    # 2. Create highlight mask (only bright areas scatter)
    highlight_mask = np.clip((luminance - threshold) / (1.0 - threshold + 1e-6), 0, 1)

    # 3. Extract the red channel of highlights only
    red_highlights = image[:, :, 0] * highlight_mask

    # 4. Blur the red highlights (simulates physical light scatter)
    ksize = radius * 2 + 1
    red_scattered = cv2.GaussianBlur(red_highlights, (ksize, ksize), 0)

    # 5. Screen blend: result = 1 - (1 - base) * (1 - overlay)
    # This ensures additive glow without exceeding 1.0
    result = image.copy()
    overlay = red_scattered * intensity
    result[:, :, 0] = 1.0 - (1.0 - result[:, :, 0]) * (1.0 - overlay)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# SOTA Color Grading Engine V2
# ────────────────────────────────────────────────────────────────

class SOTAColorEngine:
    """
    State-of-the-art color grading engine.
    Operates entirely in float32 / Oklab space.
    Uses the same PRESETS dictionary as the legacy engine.
    """

    def __init__(self, style: str = "natural", strength: float = 1.0):
        self.strength = np.clip(strength, 0.0, 1.5)
        self.preset = PRESETS.get(style, PRESETS["natural"])

    # ── Public API ──────────────────────────────────────────────

    def apply_style(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full SOTA style preset pipeline.

        Accepts both uint8 [0-255] and float32 [0-1] input.
        Always returns uint8 [0-255] for file saving compatibility.
        """
        # 1. Normalize to float32 [0-1]
        img = self._normalize_input(image)

        # 2. sRGB → Linear (gamma decode)
        #    ALL exposure/lighting math happens in linear space.
        linear = srgb_to_linear(img)

        # 3. White Balance (Temperature / Tint) — in linear space
        linear = self._adjust_temperature_linear(linear, self.preset.temperature_shift * self.strength)
        linear = self._adjust_tint_linear(linear, self.preset.tint_shift * self.strength)

        # 4. Exposure — in linear space (mathematically correct)
        linear = self._adjust_exposure_linear(linear, self.preset.exposure_offset * self.strength)

        # 5. Linear → Oklab (perceptual space for all color work)
        #    We go Linear → sRGB → Oklab (the Oklab matrices expect linear sRGB)
        srgb_mid = linear_to_srgb(np.clip(linear, 0.0, 1.0))
        oklab = srgb_to_oklab(srgb_mid)

        # 6. Tone Curve (cubic spline on Oklab L channel)
        oklab = self._apply_tone_curve_oklab(oklab, self.preset.tone_curve)

        # 7. Contrast (S-curve on Oklab L channel)
        oklab = self._adjust_contrast_oklab(oklab, self.preset.contrast)

        # 8. Per-Channel Hue/Saturation/Lightness (Oklab-based)
        oklab = self._apply_per_channel_oklab(oklab, self.preset.per_channel)

        # 9. Split Toning (Oklab a,b channels)
        oklab = self._apply_split_tone_oklab(oklab, self.preset.split_tone)

        # 10. Saturation (subtractive) & Vibrance
        oklab = self._adjust_saturation_oklab(oklab, self.preset.saturation_scale)
        oklab = self._adjust_vibrance_oklab(oklab, self.preset.vibrance_scale)

        # 11. Oklab → sRGB
        img = oklab_to_srgb(oklab)

        # 12. Vignette (in sRGB space, it's a darkening effect)
        if self.preset.vignette_strength > 0:
            img = self._apply_vignette(img, self.preset.vignette_strength,
                                       self.preset.vignette_radius)

        # 13. Grain
        if self.preset.grain_amount > 0:
            img = self._apply_grain(img, self.preset.grain_amount,
                                    self.preset.grain_size)

        # 14. Final output: float32 → uint8
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # ── Input Normalization ─────────────────────────────────────

    def _normalize_input(self, image: np.ndarray) -> np.ndarray:
        """Normalize any input to float32 [0-1]."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype in (np.float32, np.float64):
            return np.clip(image, 0.0, 1.0).astype(np.float32)
        else:
            return image.astype(np.float32) / 255.0

    # ── Temperature & Tint (Linear Space) ───────────────────────

    def _adjust_temperature_linear(self, linear: np.ndarray, shift: float) -> np.ndarray:
        """Adjust temperature in linear space. Positive = warm."""
        if abs(shift) < 0.5:
            return linear
        # In linear space, temperature shifts are multiplicative, not additive.
        # This is more physically accurate.
        warm_factor = 1.0 + shift * 0.008  # Gentle per-channel scaling
        cool_factor = 1.0 - shift * 0.006
        result = linear.copy()
        result[:, :, 0] = linear[:, :, 0] * warm_factor  # Red
        result[:, :, 2] = linear[:, :, 2] * cool_factor  # Blue
        return np.clip(result, 0.0, None)  # Allow >1.0 in linear (HDR headroom)

    def _adjust_tint_linear(self, linear: np.ndarray, shift: float) -> np.ndarray:
        """Adjust tint in linear space. Positive = magenta."""
        if abs(shift) < 0.5:
            return linear
        factor = 1.0 - shift * 0.006
        result = linear.copy()
        result[:, :, 1] = linear[:, :, 1] * factor  # Green channel
        return np.clip(result, 0.0, None)

    # ── Exposure (Linear Space) ─────────────────────────────────

    def _adjust_exposure_linear(self, linear: np.ndarray, offset: float) -> np.ndarray:
        """Adjust exposure in stops. +1 = double brightness. Done in linear = correct."""
        if abs(offset) < 0.01:
            return linear
        factor = 2.0 ** offset
        return linear * factor  # No clipping — allow HDR headroom

    # ── Tone Curve (Oklab L Channel) ────────────────────────────

    def _apply_tone_curve_oklab(self, oklab: np.ndarray, tc: ToneCurve) -> np.ndarray:
        """Apply tone curve via cubic spline to Oklab L channel."""
        # Build spline nodes from the legacy ToneCurve parameters
        nodes = self._tc_params_to_nodes(tc)
        spline = SplineToneCurve(nodes=nodes)
        lut = spline.generate_lut(size=4096)

        # Map L channel (0-1) to LUT indices
        L = oklab[:, :, 0]
        indices = np.clip(L * 4095, 0, 4095).astype(np.int32)
        oklab[:, :, 0] = lut[indices]
        return oklab

    def _tc_params_to_nodes(self, tc: ToneCurve) -> List[Tuple[float, float]]:
        """Convert legacy ToneCurve params to cubic spline control nodes."""
        # Normalize legacy values to [0, 1]
        shadow_lift = tc.shadows_lift / 255.0 * self.strength
        highlight_target = 1.0 + (tc.highlights_roll / 255.0 * self.strength)
        highlight_target = min(highlight_target, 1.0)

        # Build 5-point curve: black point, shadow, midtone, highlight, white point
        gamma = tc.midtone_gamma
        mid_out = 0.5 ** (1.0 / max(gamma, 0.01)) if abs(gamma - 1.0) > 0.01 else 0.5

        return [
            (0.0, shadow_lift),
            (0.25, shadow_lift + (mid_out - shadow_lift) * 0.4),
            (0.5, mid_out),
            (0.75, mid_out + (highlight_target - mid_out) * 0.6),
            (1.0, highlight_target),
        ]

    # ── Contrast (Oklab L Channel) ──────────────────────────────

    def _adjust_contrast_oklab(self, oklab: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast on Oklab L channel (perceptually uniform)."""
        effective = 1.0 + (factor - 1.0) * self.strength
        if abs(effective - 1.0) < 0.01:
            return oklab
        L = oklab[:, :, 0]
        mid = 0.5  # Oklab L midpoint
        oklab[:, :, 0] = np.clip(mid + (L - mid) * effective, 0.0, 1.0)
        return oklab

    # ── Per-Channel HSL (Oklab-based) ───────────────────────────

    def _apply_per_channel_oklab(self, oklab: np.ndarray,
                                  channels: Dict[str, ChannelHSL]) -> np.ndarray:
        """Apply per-channel hue/saturation/lightness adjustments in Oklab."""
        if not channels:
            return oklab

        # We need hue angle to identify color channels
        hue = oklab_hue(oklab)  # radians, [-π, π]
        chroma = oklab_chroma(oklab)

        # Convert Oklab hue (radians) to approximate OpenCV-style hue (0-180)
        # for compatibility with existing HSL_RANGES definitions
        hue_deg = np.degrees(hue) % 360  # [0, 360]
        hue_cv = hue_deg / 2.0  # [0, 180] to match OpenCV convention

        for color_name, adj in channels.items():
            if abs(adj.hue) < 0.5 and abs(adj.sat) < 0.5 and abs(adj.lum) < 0.5:
                continue

            ranges = HSL_RANGES.get(color_name, [])
            mask = np.zeros(hue_cv.shape, dtype=bool)
            for (lo, hi) in ranges:
                mask |= (hue_cv >= lo) & (hue_cv <= hi)

            # Require some chroma (exclude greys)
            mask &= (chroma > 0.02)

            if not mask.any():
                continue

            # Lightness adjustment
            if abs(adj.lum) >= 0.5:
                oklab[:, :, 0][mask] += adj.lum * self.strength * 0.005

            # Saturation (chroma) adjustment
            if abs(adj.sat) >= 0.5:
                chroma_scale = 1.0 + adj.sat * self.strength * 0.005
                oklab[:, :, 1][mask] *= chroma_scale
                oklab[:, :, 2][mask] *= chroma_scale

            # Hue rotation
            if abs(adj.hue) >= 0.5:
                hue_shift_rad = np.radians(adj.hue * self.strength * 0.5)
                a = oklab[:, :, 1][mask]
                b = oklab[:, :, 2][mask]
                cos_h = np.cos(hue_shift_rad)
                sin_h = np.sin(hue_shift_rad)
                oklab[:, :, 1][mask] = a * cos_h - b * sin_h
                oklab[:, :, 2][mask] = a * sin_h + b * cos_h

        oklab[:, :, 0] = np.clip(oklab[:, :, 0], 0.0, 1.0)
        return oklab

    # ── Split Toning (Oklab a,b) ────────────────────────────────

    def _apply_split_tone_oklab(self, oklab: np.ndarray, st: SplitTone) -> np.ndarray:
        """Apply split toning in Oklab space."""
        if st.shadow_hue is None and st.highlight_hue is None:
            return oklab

        L = oklab[:, :, 0]

        # Shadow mask (strongest at L=0)
        shadow_mask = np.clip(1.0 - L * 2, 0, 1)
        # Highlight mask (strongest at L=1)
        highlight_mask = np.clip(L * 2 - 1, 0, 1)

        if st.shadow_hue is not None and st.shadow_saturation > 0:
            rad = np.radians(st.shadow_hue)
            intensity = st.shadow_saturation * self.strength * 0.002
            oklab[:, :, 1] += shadow_mask * np.cos(rad) * intensity
            oklab[:, :, 2] += shadow_mask * np.sin(rad) * intensity

        if st.highlight_hue is not None and st.highlight_saturation > 0:
            rad = np.radians(st.highlight_hue)
            intensity = st.highlight_saturation * self.strength * 0.002
            oklab[:, :, 1] += highlight_mask * np.cos(rad) * intensity
            oklab[:, :, 2] += highlight_mask * np.sin(rad) * intensity

        return oklab

    # ── Saturation & Vibrance (Oklab) ───────────────────────────

    def _adjust_saturation_oklab(self, oklab: np.ndarray, scale: float) -> np.ndarray:
        """Scale saturation (chroma) uniformly in Oklab, with subtractive density."""
        effective = 1.0 + (scale - 1.0) * self.strength
        if abs(effective - 1.0) < 0.01:
            return oklab
        return subtractive_saturate(oklab, factor=effective)

    def _adjust_vibrance_oklab(self, oklab: np.ndarray, scale: float) -> np.ndarray:
        """Boost under-saturated colors more than saturated ones (in Oklab)."""
        effective = 1.0 + (scale - 1.0) * self.strength
        if abs(effective - 1.0) < 0.01:
            return oklab

        chroma = oklab_chroma(oklab)
        # Weight: low-chroma pixels get a stronger boost
        max_chroma = chroma.max() + 1e-6
        chroma_norm = chroma / max_chroma
        weight = 1.0 - chroma_norm  # Higher for desaturated
        per_pixel_factor = 1.0 + (effective - 1.0) * weight

        oklab[:, :, 1] *= per_pixel_factor
        oklab[:, :, 2] *= per_pixel_factor
        return oklab

    # ── Vignette ────────────────────────────────────────────────

    def _apply_vignette(self, img: np.ndarray, strength: float,
                        radius: float) -> np.ndarray:
        """Radial darkening from center (in sRGB float32)."""
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist_norm = dist / max_dist
        mask = 1.0 - np.clip((dist_norm - radius) / (1.0 - radius + 1e-5), 0, 1)
        mask = mask ** 2
        vignette = 1.0 - strength * self.strength * (1.0 - mask)
        return img * vignette[:, :, np.newaxis]

    # ── Grain ───────────────────────────────────────────────────

    def _apply_grain(self, img: np.ndarray, amount: float,
                     size: float) -> np.ndarray:
        """Add film grain scaled by luminance (heavier in shadows/midtones)."""
        h, w = img.shape[:2]
        # Generate at reduced resolution for larger grain structure
        gh, gw = max(int(h / size), 1), max(int(w / size), 1)
        noise = np.random.normal(0, 1, (gh, gw)).astype(np.float32)

        # Resize to full resolution
        import cv2
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)

        # Luminance-weighted: grain is heavier in shadows/midtones, absent in highlights
        luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        grain_weight = 1.0 - np.clip(luminance, 0, 1)  # Stronger in darks
        grain_weight = grain_weight ** 0.5  # Soften the falloff

        # Scale noise and apply
        scaled_noise = noise * grain_weight * amount * self.strength / 255.0
        result = img + scaled_noise[:, :, np.newaxis]
        return np.clip(result, 0.0, 1.0)
