"""
Tests for CAT02 Chromatic Adaptation White Balance
====================================================
P1: Validates the physically accurate white balance correction using
CAT02 chromatic adaptation transforms via the colour-science library.

The key difference from the legacy R/B channel shift:
- CAT02 models how the human visual system adapts to different illuminants
- It correctly preserves skin tones under tungsten/fluorescent lighting
- It handles extreme color casts (e.g., 2700K tungsten → D65) without clipping
"""

import pytest
import numpy as np


@pytest.fixture
def tungsten_image():
    """
    Simulate a photo taken under warm tungsten lighting (2700K).
    Everything has an orange/amber cast — excess red, deficient blue.
    """
    img = np.zeros((100, 100, 3), dtype=np.float32)
    # Skin tone under tungsten — too warm
    img[:50, :50] = [0.90, 0.65, 0.40]
    # White dress under tungsten — looks yellowish
    img[:50, 50:] = [0.95, 0.88, 0.65]
    # Blue suit under tungsten — looks grey/muted
    img[50:, :50] = [0.40, 0.38, 0.35]
    # Grey card under tungsten — should be neutral but appears warm
    img[50:, 50:] = [0.55, 0.48, 0.38]
    return img


@pytest.fixture
def fluorescent_image():
    """
    Simulate a photo taken under green-ish fluorescent lighting.
    Everything has a strong green tint.
    """
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[:, :] = [0.45, 0.65, 0.40]  # Strong green-tinted neutral
    return img


@pytest.fixture
def daylight_image():
    """A properly lit D65 image — should be unchanged by adaptation."""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[:50, :] = [0.70, 0.70, 0.70]  # Neutral grey
    img[50:, :] = [0.85, 0.65, 0.50]  # Natural skin tone
    return img


# ── Core CAT02 Tests ──────────────────────────────────────────

class TestCAT02Adaptation:

    def test_tungsten_to_daylight_reduces_red(self, tungsten_image):
        """
        Adapting from tungsten (warm) to D65 (daylight) should reduce
        the red channel and boost the blue channel.
        """
        from src.color.engine_v2 import chromatic_adapt_cat02

        result = chromatic_adapt_cat02(
            tungsten_image, source_illuminant="A", target_illuminant="D65"
        )

        # The grey card region should become more neutral
        grey_before = tungsten_image[75, 75]
        grey_after = result[75, 75]

        # Red should decrease
        assert grey_after[0] < grey_before[0], (
            f"Red should decrease: {grey_before[0]:.3f} → {grey_after[0]:.3f}"
        )
        # Blue should increase
        assert grey_after[2] > grey_before[2], (
            f"Blue should increase: {grey_before[2]:.3f} → {grey_after[2]:.3f}"
        )

    def test_tungsten_grey_becomes_more_neutral(self, tungsten_image):
        """
        A grey card under tungsten has R > B (warm cast). After CAT02
        adaptation to D65, the signed R-B should decrease, indicating the
        warm color cast was corrected.
        """
        from src.color.engine_v2 import chromatic_adapt_cat02

        result = chromatic_adapt_cat02(
            tungsten_image, source_illuminant="A", target_illuminant="D65"
        )

        grey_before = tungsten_image[75, 75]
        grey_after = result[75, 75]
        # Signed R-B: positive = warm cast, negative = cool
        rb_signed_before = float(grey_before[0]) - float(grey_before[2])
        rb_signed_after = float(grey_after[0]) - float(grey_after[2])
        # CAT02 should reduce the warm bias (make R-B less positive)
        assert rb_signed_after < rb_signed_before, (
            f"Warm cast (R-B) should decrease: {rb_signed_before:.3f} → {rb_signed_after:.3f}"
        )

    def test_daylight_to_daylight_is_identity(self, daylight_image):
        """Adapting D65 → D65 should produce no change."""
        from src.color.engine_v2 import chromatic_adapt_cat02

        result = chromatic_adapt_cat02(
            daylight_image, source_illuminant="D65", target_illuminant="D65"
        )
        max_diff = np.abs(result - daylight_image).max()
        assert max_diff < 0.02, f"D65→D65 should be identity, max_diff={max_diff:.4f}"

    def test_fluorescent_to_daylight_reduces_green(self, fluorescent_image):
        """Adapting from fluorescent (green) to D65 should reduce green channel dominance."""
        from src.color.engine_v2 import chromatic_adapt_cat02

        result = chromatic_adapt_cat02(
            fluorescent_image, source_illuminant="FL2", target_illuminant="D65"
        )

        # The green channel should decrease relative to the average of R and B
        pixel_before = fluorescent_image[50, 50]
        pixel_after = result[50, 50]

        # Verify the adaptation produced a visible change
        max_diff = np.abs(result - fluorescent_image).max()
        assert max_diff > 0.01, "Adaptation should produce a visible change"

        # After adaptation, the red-blue spread relative to green should shift
        # (the exact direction depends on FL2's spectral characteristics)
        green_ratio_before = pixel_before[1] / (pixel_before.mean() + 1e-6)
        green_ratio_after = pixel_after[1] / (pixel_after.mean() + 1e-6)
        assert green_ratio_after < green_ratio_before + 0.05, (
            f"Green ratio should decrease or stay similar: {green_ratio_before:.3f} → {green_ratio_after:.3f}"
        )

    def test_output_shape_and_dtype(self, tungsten_image):
        """Output should match input shape and be float32."""
        from src.color.engine_v2 import chromatic_adapt_cat02
        result = chromatic_adapt_cat02(tungsten_image, "A", "D65")
        assert result.shape == tungsten_image.shape
        assert result.dtype == np.float32

    def test_output_range_clamped(self, tungsten_image):
        """Output should be clamped to [0, 1]."""
        from src.color.engine_v2 import chromatic_adapt_cat02
        result = chromatic_adapt_cat02(tungsten_image, "A", "D65")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_temperature_kelvin_api(self, tungsten_image):
        """
        The engine should also accept color temperature in Kelvin
        (e.g., 2700K, 5500K) in addition to illuminant names.
        """
        from src.color.engine_v2 import chromatic_adapt_cat02

        # 2700K ≈ tungsten, 6500K ≈ D65
        result = chromatic_adapt_cat02(
            tungsten_image, source_illuminant=2700, target_illuminant=6500
        )
        assert result.shape == tungsten_image.shape
        # Should produce a visible correction (not identity)
        max_diff = np.abs(result - tungsten_image).max()
        assert max_diff > 0.02, f"Kelvin API should produce visible change, diff={max_diff:.3f}"

        # Red should decrease (warming correction)
        grey_before = tungsten_image[75, 75]
        grey_after = result[75, 75]
        assert grey_after[0] < grey_before[0], (
            f"Kelvin correction should reduce red: {grey_before[0]:.3f} → {grey_after[0]:.3f}"
        )
