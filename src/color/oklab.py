"""
Oklab Color Space Module (SOTA Engine Foundation)
==================================================
Pure NumPy implementation of sRGB ↔ Linear ↔ Oklab transforms.

Oklab is a perceptually uniform color space designed by Björn Ottosson.
It solves the well-known hue-shift and luminance-skew problems of CIELAB
and HSV, making it ideal for color grading operations where hue and
saturation adjustments must not interfere with perceived brightness.

Reference: https://bottosson.github.io/posts/oklab/

All functions operate on float32 arrays in [0.0, 1.0] range for sRGB,
and produce float32 Oklab values where:
  - L: Lightness  (0.0 = black, 1.0 = white)
  - a: Green←→Red axis
  - b: Blue←→Yellow axis
"""

import numpy as np


# ── sRGB ↔ Linear Transforms ───────────────────────────────────
# The sRGB standard uses a piecewise gamma curve:
#   linear = srgb / 12.92                      if srgb <= 0.04045
#   linear = ((srgb + 0.055) / 1.055) ^ 2.4    if srgb > 0.04045

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB gamma-encoded values to linear light.

    Args:
        srgb: float32 array in [0.0, 1.0] range, shape (..., 3).

    Returns:
        float32 array in linear light space.
    """
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    # Piecewise sRGB transfer function
    low = srgb / 12.92
    high = np.power((srgb + 0.055) / 1.055, 2.4)
    return np.where(srgb <= 0.04045, low, high).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Convert linear light values to sRGB gamma-encoded.

    Args:
        linear: float32 array in linear light space.

    Returns:
        float32 array in [0.0, 1.0] sRGB range.
    """
    linear = np.clip(linear, 0.0, 1.0).astype(np.float32)
    low = linear * 12.92
    high = 1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    return np.where(linear <= 0.0031308, low, high).astype(np.float32)


# ── Linear sRGB → Oklab ────────────────────────────────────────
# The Oklab transform is a two-step matrix multiplication:
#   1. Linear sRGB → LMS (cone response)
#   2. LMS^(1/3) → Oklab (L, a, b)
#
# Matrices are from Björn Ottosson's original derivation.

# Step 1: Linear sRGB → LMS
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float64)

# Step 2: LMS^(1/3) → Oklab
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float64)

# Inverse matrices (for Oklab → sRGB)
_M2_inv = np.linalg.inv(_M2)
_M1_inv = np.linalg.inv(_M1)


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB (gamma-encoded, 0–1 float32) to Oklab.

    Args:
        srgb: float32 array of shape (H, W, 3) in [0.0, 1.0].

    Returns:
        float32 array of shape (H, W, 3) in Oklab [L, a, b].
        L is in [0, 1], a and b are approximately in [-0.5, 0.5].
    """
    linear = srgb_to_linear(srgb)

    # Reshape for matrix multiplication: (H*W, 3)
    shape = linear.shape
    pixels = linear.reshape(-1, 3).astype(np.float64)

    # Step 1: Linear sRGB → LMS
    lms = pixels @ _M1.T

    # Cube root (handle negative values safely for out-of-gamut)
    lms_g = np.sign(lms) * np.abs(lms) ** (1.0 / 3.0)

    # Step 2: LMS^(1/3) → Oklab
    oklab = lms_g @ _M2.T

    return oklab.reshape(shape).astype(np.float32)


def oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    """
    Convert Oklab to sRGB (gamma-encoded, 0–1 float32).

    Args:
        oklab: float32 array of shape (H, W, 3) in Oklab [L, a, b].

    Returns:
        float32 array of shape (H, W, 3) in sRGB [0.0, 1.0].
    """
    shape = oklab.shape
    pixels = oklab.reshape(-1, 3).astype(np.float64)

    # Inverse Step 2: Oklab → LMS^(1/3)
    lms_g = pixels @ _M2_inv.T

    # Cube: LMS^(1/3) → LMS
    lms = lms_g ** 3

    # Inverse Step 1: LMS → Linear sRGB
    linear = lms @ _M1_inv.T

    linear = linear.reshape(shape).astype(np.float32)
    return linear_to_srgb(np.clip(linear, 0.0, 1.0))


# ── Oklab Utilities ─────────────────────────────────────────────

def oklab_chroma(oklab: np.ndarray) -> np.ndarray:
    """
    Compute chroma (perceptual saturation) from Oklab values.
    Chroma = sqrt(a² + b²).

    Args:
        oklab: float32 array of shape (H, W, 3).

    Returns:
        float32 array of shape (H, W) with chroma values.
    """
    a = oklab[:, :, 1]
    b = oklab[:, :, 2]
    return np.sqrt(a ** 2 + b ** 2).astype(np.float32)


def oklab_hue(oklab: np.ndarray) -> np.ndarray:
    """
    Compute hue angle (in radians) from Oklab values.
    Hue = atan2(b, a).

    Args:
        oklab: float32 array of shape (H, W, 3).

    Returns:
        float32 array of shape (H, W) with hue angles in [−π, π].
    """
    a = oklab[:, :, 1]
    b = oklab[:, :, 2]
    return np.arctan2(b, a).astype(np.float32)
