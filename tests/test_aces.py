"""
Tests for P5: ACES/OCIO Color Management Pipeline
===================================================
Validates the ACES color space transforms using OpenColorIO:
  - sRGB → ACEScg (scene-linear working space)
  - ACEScg → sRGB (output display transform)
  - ACEScg roundtrip accuracy
"""

import pytest
import numpy as np


@pytest.fixture
def srgb_image():
    """A varied sRGB image for ACES conversion testing."""
    rng = np.random.RandomState(42)
    return rng.rand(50, 50, 3).astype(np.float32)


@pytest.fixture
def neutral_grey():
    """Mid-grey sRGB — should map to scene-linear mid-grey in ACEScg."""
    return np.full((20, 20, 3), 0.5, dtype=np.float32)


@pytest.fixture
def clipped_image():
    """Image with values near 0 and 1 (edge cases for ACES transform)."""
    img = np.zeros((20, 20, 3), dtype=np.float32)
    img[:10, :] = 0.01  # Near black
    img[10:, :] = 0.99  # Near white
    return img


# ── ACES Transforms ───────────────────────────────────────────

class TestACESTransforms:

    def test_srgb_to_acescg_produces_linear(self, neutral_grey):
        """sRGB 0.5 (gamma) should become ~0.18 in ACEScg (scene-linear)."""
        from src.color.aces import srgb_to_acescg
        acescg = srgb_to_acescg(neutral_grey)
        # sRGB 0.5 → linear ~0.214 → ACEScg mid-grey ~0.18–0.22
        mid_val = acescg[10, 10, 0]
        assert 0.10 < mid_val < 0.30, f"Mid-grey should be ~0.18 in ACEScg, got {mid_val:.3f}"

    def test_acescg_roundtrip(self, srgb_image):
        """sRGB → ACEScg → sRGB should be identity within tolerance."""
        from src.color.aces import srgb_to_acescg, acescg_to_srgb
        acescg = srgb_to_acescg(srgb_image)
        roundtrip = acescg_to_srgb(acescg)
        max_error = np.abs(roundtrip - srgb_image).max()
        assert max_error < 0.01, f"Roundtrip error too high: {max_error:.4f}"

    def test_acescg_is_float32(self, srgb_image):
        """ACEScg output should be float32."""
        from src.color.aces import srgb_to_acescg
        acescg = srgb_to_acescg(srgb_image)
        assert acescg.dtype == np.float32

    def test_acescg_shape_preserved(self, srgb_image):
        """Output shape should match input."""
        from src.color.aces import srgb_to_acescg
        acescg = srgb_to_acescg(srgb_image)
        assert acescg.shape == srgb_image.shape

    def test_black_stays_black(self):
        """Pure black should remain black in ACEScg."""
        from src.color.aces import srgb_to_acescg
        black = np.zeros((5, 5, 3), dtype=np.float32)
        result = srgb_to_acescg(black)
        np.testing.assert_allclose(result, 0.0, atol=0.001)

    def test_acescg_white_is_scene_linear(self):
        """sRGB white (1.0) should map to ACEScg ~1.0 (scene-referred)."""
        from src.color.aces import srgb_to_acescg
        white = np.ones((5, 5, 3), dtype=np.float32)
        result = srgb_to_acescg(white)
        # ACEScg white should be approximately 1.0 (within gamut mapping tolerance)
        assert result[2, 2, 0] > 0.8, f"White should be bright in ACEScg: {result[2, 2, 0]:.3f}"

    def test_edge_values_no_nan(self, clipped_image):
        """Near-zero and near-one values should not produce NaN/Inf."""
        from src.color.aces import srgb_to_acescg, acescg_to_srgb
        acescg = srgb_to_acescg(clipped_image)
        assert not np.any(np.isnan(acescg)), "NaN detected in ACEScg"
        assert not np.any(np.isinf(acescg)), "Inf detected in ACEScg"
        srgb = acescg_to_srgb(acescg)
        assert not np.any(np.isnan(srgb)), "NaN detected in reconstructed sRGB"
