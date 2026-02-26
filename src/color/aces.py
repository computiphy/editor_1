"""
ACES / OpenColorIO Color Management
=====================================
Provides sRGB ↔ ACEScg (AP1 scene-linear) transforms using the
official ACES color matrices.

Why ACES?
  - ACES (Academy Color Encoding System) is the film industry standard
    for color management. It defines a scene-linear working space that
    can represent every visible color (and beyond for HDR).
  - Working in ACEScg eliminates gamut clipping artifacts that occur
    when manipulating saturated colors in sRGB.
  - All VFX-grade tools (Nuke, DaVinci, Baselight) use ACES internally.

Pipeline integration:
  - Use srgb_to_acescg() at pipeline entry (before any grading)
  - Grade entirely in ACEScg (scene-linear, wide-gamut)
  - Use acescg_to_srgb() at pipeline exit (for display/file output)

Note: This module uses the pure-matrix approach (AP0 → AP1 → sRGB)
rather than requiring a full OCIO config file. For studios that need
full OCIO config support, see the ocio_pipeline() function.
"""

import numpy as np
from typing import Optional

from src.color.oklab import srgb_to_linear, linear_to_srgb


# ────────────────────────────────────────────────────────────────
# ACES Color Matrices
# ────────────────────────────────────────────────────────────────

# sRGB (D65) → XYZ (D65) from IEC 61966-2-1
_M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# XYZ (D65) → sRGB (D65)
_M_XYZ_TO_SRGB = np.linalg.inv(_M_SRGB_TO_XYZ)

# XYZ (D65) → ACEScg (AP1, D60-referenced, with D65→D60 Bradford CAT baked in)
# This is the combined matrix from the ACES spec (S-2014-004)
_M_XYZ_D65_TO_AP1 = np.array([
    [ 1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587,  1.6153315917,  0.0167563477],
    [ 0.0117218943, -0.0082844420,  0.9883948585],
], dtype=np.float64)

# ACEScg (AP1) → XYZ (D65)
_M_AP1_TO_XYZ_D65 = np.linalg.inv(_M_XYZ_D65_TO_AP1)

# Combined: sRGB linear → ACEScg
_M_SRGB_TO_ACESCG = _M_XYZ_D65_TO_AP1 @ _M_SRGB_TO_XYZ

# Combined: ACEScg → sRGB linear
_M_ACESCG_TO_SRGB = np.linalg.inv(_M_SRGB_TO_ACESCG)


# ────────────────────────────────────────────────────────────────
# Transform Functions
# ────────────────────────────────────────────────────────────────

def srgb_to_acescg(image: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image to ACEScg (AP1 scene-linear).

    Pipeline:
      1. sRGB gamma → Linear sRGB (using the standard piecewise transfer)
      2. Linear sRGB → ACEScg via combined 3x3 matrix

    Args:
        image: float32 array (H, W, 3) in sRGB [0-1].

    Returns:
        float32 array (H, W, 3) in ACEScg scene-linear.
        Note: Values can exceed 1.0 (scene-referred, not display-referred).
    """
    # 1. Gamma decode
    linear = srgb_to_linear(image)

    # 2. Matrix multiply: linear sRGB → ACEScg
    shape = linear.shape
    pixels = linear.reshape(-1, 3).astype(np.float64)
    acescg = pixels @ _M_SRGB_TO_ACESCG.T
    acescg = acescg.reshape(shape)

    return acescg.astype(np.float32)


def acescg_to_srgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an ACEScg image back to sRGB.

    Pipeline:
      1. ACEScg → Linear sRGB via inverse matrix
      2. Linear sRGB → sRGB gamma (using the standard piecewise transfer)
      3. Clamp to [0, 1] (gamut mapping — ACEScg colors outside sRGB are clipped)

    Args:
        image: float32 array (H, W, 3) in ACEScg scene-linear.

    Returns:
        float32 array (H, W, 3) in sRGB [0-1].
    """
    # 1. Matrix multiply: ACEScg → linear sRGB
    shape = image.shape
    pixels = image.reshape(-1, 3).astype(np.float64)
    linear = pixels @ _M_ACESCG_TO_SRGB.T

    # 2. Clamp negatives (out-of-gamut colors)
    linear = np.clip(linear.reshape(shape), 0.0, None).astype(np.float32)

    # 3. Gamma encode
    srgb = linear_to_srgb(linear)

    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# OpenColorIO Pipeline (Optional)
# ────────────────────────────────────────────────────────────────

def ocio_transform(image: np.ndarray,
                   src_colorspace: str = "ACES - ACEScg",
                   dst_colorspace: str = "Output - sRGB",
                   config_path: Optional[str] = None) -> np.ndarray:
    """
    Apply an OpenColorIO color space transform.

    This function uses PyOpenColorIO for studios that require full OCIO
    config support (custom color spaces, CDLs, display transforms, etc.).

    Args:
        image: float32 array (H, W, 3).
        src_colorspace: Source color space name (as defined in OCIO config).
        dst_colorspace: Destination color space name.
        config_path: Path to OCIO config file. If None, uses ACES default.

    Returns:
        float32 array (H, W, 3) in target color space.
    """
    try:
        import PyOpenColorIO as ocio
    except ImportError:
        raise ImportError(
            "PyOpenColorIO is required for OCIO transforms. "
            "Install with: pip install opencolorio"
        )

    if config_path:
        config = ocio.Config.CreateFromFile(config_path)
    else:
        config = ocio.Config.CreateFromBuiltinConfig("studio-config-v2.1.0_aces-v1.3_ocio-v2.3")

    processor = config.getProcessor(src_colorspace, dst_colorspace)
    cpu = processor.getDefaultCPUProcessor()

    # OCIO expects a flat float32 buffer
    h, w = image.shape[:2]
    buf = image.astype(np.float32).copy()

    # Apply in-place
    img_desc = ocio.PackedImageDesc(
        buf,
        w, h,
        3,  # num channels
    )
    cpu.apply(img_desc)

    return buf
