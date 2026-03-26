"""
Log Curve Transfer Functions
=============================
Converts between linear-light scene data and camera-log encodings.

Many professional 3D LUTs are designed to accept log-encoded input
(e.g. S-Log3 footage from Sony, LogC from ARRI). When the source
image is linear (e.g. developed from RAW), it must be "logified"
before the LUT will produce correct results.

Supported curves
----------------
slog3     : Sony S-Log3  (mid-grey 0.18 → ~0.41)
logc3     : ARRI LogC3   (ALEXA EI 800, mid-grey 0.18 → ~0.39)
logc4     : ARRI LogC4   (ALEXA 35)
cineon    : Kodak Cineon  (printing density log)

All functions operate on float32 arrays in [0–1] range.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Sony S-Log3  (Sony Technical Specification)
# ═══════════════════════════════════════════════════════════════════
# Forward:
#   x >= 0.01125000 :  y = (420 + log10((x+0.01)/(0.18+0.01)) * 261.5) / 1023
#   x <  0.01125000 :  y = (x*(171.2102946929 - 95)/0.01125000 + 95) / 1023
#
# 0.18 maps to 420/1023 ≈ 0.41

_SLOG3_CUT = 0.01125000


def linear_to_slog3(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light [0-1] to Sony S-Log3 code values [0-1]."""
    x = np.clip(linear, 0.0, None).astype(np.float64)
    out = np.empty_like(x)

    low = x < _SLOG3_CUT
    out[low] = (x[low] * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0

    hi = ~low
    out[hi] = (420.0 + np.log10((x[hi] + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def slog3_to_linear(slog3: np.ndarray) -> np.ndarray:
    """Convert Sony S-Log3 code values [0-1] back to linear-light [0-1]."""
    y = np.clip(slog3, 0.0, 1.0).astype(np.float64)
    out = np.empty_like(y)

    y_cut = (_SLOG3_CUT * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0
    low = y < y_cut
    out[low] = (y[low] * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0)

    hi = ~low
    out[hi] = (0.18 + 0.01) * 10.0 ** ((y[hi] * 1023.0 - 420.0) / 261.5) - 0.01

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# ARRI LogC3 (ALEXA, EI 800)
# Reference: ARRI LogC curve — SUP 3.x / ALF-2
# ═══════════════════════════════════════════════════════════════════

_LOGC3_CUT  = 0.010591
_LOGC3_A    = 5.555556
_LOGC3_B    = 0.052272
_LOGC3_C    = 0.247190
_LOGC3_D    = 0.385537
_LOGC3_E    = 5.367655
_LOGC3_F    = 0.092809


def linear_to_logc3(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light [0-1] to ARRI LogC3 code values [0-1]."""
    x = np.clip(linear, 0.0, None).astype(np.float64)
    out = np.empty_like(x)

    low = x < _LOGC3_CUT
    out[low]  = _LOGC3_E * x[low] + _LOGC3_F
    out[~low] = _LOGC3_C * np.log10(_LOGC3_A * x[~low] + _LOGC3_B) + _LOGC3_D

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def logc3_to_linear(logc3: np.ndarray) -> np.ndarray:
    """Convert ARRI LogC3 code values [0-1] back to linear-light [0-1]."""
    y = np.clip(logc3, 0.0, 1.0).astype(np.float64)
    out = np.empty_like(y)

    y_cut = _LOGC3_E * _LOGC3_CUT + _LOGC3_F
    low = y < y_cut
    out[low]  = (y[low] - _LOGC3_F) / _LOGC3_E
    out[~low] = (10.0 ** ((y[~low] - _LOGC3_D) / _LOGC3_C) - _LOGC3_B) / _LOGC3_A

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# ARRI LogC4 (ALEXA 35)
# Reference: ARRI LogC4 Specification Rev 1.0
# ═══════════════════════════════════════════════════════════════════

_LOGC4_A = 2231.826309067688
_LOGC4_B = 64.0
_LOGC4_C = 0.0740718950408889
_LOGC4_S = 7.0
_LOGC4_T = 0.01011361589284066


def linear_to_logc4(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light [0-1] to ARRI LogC4 code values [0-1]."""
    x = np.clip(linear, 0.0, None).astype(np.float64)
    out = np.empty_like(x)

    low = x < _LOGC4_T
    t_log = _LOGC4_C * np.log2(_LOGC4_A * _LOGC4_T + _LOGC4_B) + _LOGC4_C
    out[low]  = (x[low] - _LOGC4_T) / _LOGC4_S + t_log
    out[~low] = _LOGC4_C * np.log2(_LOGC4_A * x[~low] + _LOGC4_B) + _LOGC4_C

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def logc4_to_linear(logc4: np.ndarray) -> np.ndarray:
    """Convert ARRI LogC4 code values [0-1] back to linear-light [0-1]."""
    y = np.clip(logc4, 0.0, 1.0).astype(np.float64)
    out = np.empty_like(y)

    y_cut = _LOGC4_C * np.log2(_LOGC4_A * _LOGC4_T + _LOGC4_B) + _LOGC4_C
    low = y < y_cut
    out[low]  = (y[low] - y_cut) * _LOGC4_S + _LOGC4_T
    out[~low] = (2.0 ** ((y[~low] - _LOGC4_C) / _LOGC4_C) - _LOGC4_B) / _LOGC4_A

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Kodak Cineon / DPX Log
# Reference: Kodak 10-bit print density mapping
#   offset (black) = 95/1023,  white = 685/1023,  gamma = 0.6
# ═══════════════════════════════════════════════════════════════════

_CINEON_BLACK = 95.0 / 1023.0
_CINEON_WHITE = 685.0 / 1023.0
_CINEON_RANGE = _CINEON_WHITE - _CINEON_BLACK


def linear_to_cineon(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light [0-1] to Cineon log [0-1]."""
    x = np.clip(linear, 1e-10, None).astype(np.float64)
    # 10-bit code value normalised to 0-1
    log_val = np.log10(x)   # log10(0..1) is in [-10, 0]
    # Map so that 1.0 → _CINEON_WHITE, ~0 → _CINEON_BLACK
    out = _CINEON_BLACK + (_CINEON_RANGE * (log_val / np.log10(1e-10))) * -1
    # Simpler form: direct power mapping
    out = _CINEON_BLACK + _CINEON_RANGE * (1.0 + log_val / 2.046)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def cineon_to_linear(cineon: np.ndarray) -> np.ndarray:
    """Convert Cineon log [0-1] back to linear-light [0-1]."""
    y = np.clip(cineon, 0.0, 1.0).astype(np.float64)
    normalised = (y - _CINEON_BLACK) / _CINEON_RANGE
    out = 10.0 ** ((normalised - 1.0) * 2.046)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Dispatcher — used by the pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════

SUPPORTED_CURVES = {
    "slog3":  (linear_to_slog3,  slog3_to_linear),
    "logc3":  (linear_to_logc3,  logc3_to_linear),
    "logc4":  (linear_to_logc4,  logc4_to_linear),
    "cineon": (linear_to_cineon, cineon_to_linear),
}


def logify(image: np.ndarray, curve: str = "slog3") -> np.ndarray:
    """
    Convert a linear-light float32 [0-1] image to log space.

    Args:
        image: float32 array (H, W, 3) in [0-1].
        curve: One of "slog3", "logc3", "logc4", "cineon".

    Returns:
        float32 array (H, W, 3) in [0-1] log-encoded.
    """
    if curve not in SUPPORTED_CURVES:
        raise ValueError(
            f"Unknown log curve '{curve}'. "
            f"Supported: {sorted(SUPPORTED_CURVES.keys())}"
        )
    to_log, _ = SUPPORTED_CURVES[curve]
    return to_log(image)


def delogify(image: np.ndarray, curve: str = "slog3") -> np.ndarray:
    """
    Convert a log-encoded float32 [0-1] image back to linear-light.

    Args:
        image: float32 array (H, W, 3) in [0-1] log-encoded.
        curve: One of "slog3", "logc3", "logc4", "cineon".

    Returns:
        float32 array (H, W, 3) in [0-1] linear-light.
    """
    if curve not in SUPPORTED_CURVES:
        raise ValueError(
            f"Unknown log curve '{curve}'. "
            f"Supported: {sorted(SUPPORTED_CURVES.keys())}"
        )
    _, to_linear = SUPPORTED_CURVES[curve]
    return to_linear(image)
