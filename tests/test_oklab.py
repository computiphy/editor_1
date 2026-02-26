"""
Tests for the Oklab Color Space Module (SOTA Engine P0)
========================================================
Validates the pure-NumPy implementation of sRGB ↔ Linear ↔ Oklab transforms.
These are the foundation of the entire SOTA color engine.

Reference: https://bottosson.github.io/posts/oklab/
"""

import pytest
import numpy as np


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def solid_red():
    """Pure sRGB red (255, 0, 0) as float32 0–1."""
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[:, :, 0] = 1.0
    return img

@pytest.fixture
def solid_white():
    """Pure sRGB white (255, 255, 255) as float32 0–1."""
    return np.ones((10, 10, 3), dtype=np.float32)

@pytest.fixture
def solid_black():
    """Pure sRGB black (0, 0, 0) as float32 0–1."""
    return np.zeros((10, 10, 3), dtype=np.float32)

@pytest.fixture
def gradient_image():
    """A 100x100 gradient with varied hues for roundtrip testing."""
    rng = np.random.RandomState(42)
    return rng.rand(100, 100, 3).astype(np.float32)


# ── sRGB ↔ Linear Conversions ──────────────────────────────────

class TestLinearConversions:

    def test_srgb_to_linear_black_is_zero(self, solid_black):
        """Black (0.0) should remain 0.0 in linear space."""
        from src.color.oklab import srgb_to_linear
        result = srgb_to_linear(solid_black)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_srgb_to_linear_white_is_one(self, solid_white):
        """White (1.0) should remain 1.0 in linear space."""
        from src.color.oklab import srgb_to_linear
        result = srgb_to_linear(solid_white)
        np.testing.assert_allclose(result, 1.0, atol=1e-7)

    def test_srgb_to_linear_midgrey(self):
        """sRGB 0.5 should convert to ~0.214 in linear (gamma correction)."""
        from src.color.oklab import srgb_to_linear
        mid = np.full((1, 1, 3), 0.5, dtype=np.float32)
        result = srgb_to_linear(mid)
        # sRGB 0.5 → linear ≈ 0.214
        np.testing.assert_allclose(result[0, 0, 0], 0.214, atol=0.01)

    def test_linear_to_srgb_roundtrip(self, gradient_image):
        """sRGB → Linear → sRGB should be identity."""
        from src.color.oklab import srgb_to_linear, linear_to_srgb
        linear = srgb_to_linear(gradient_image)
        roundtrip = linear_to_srgb(linear)
        np.testing.assert_allclose(roundtrip, gradient_image, atol=1e-5)

    def test_output_is_float32(self, solid_red):
        """Output must be float32, never uint8."""
        from src.color.oklab import srgb_to_linear
        result = srgb_to_linear(solid_red)
        assert result.dtype == np.float32


# ── sRGB ↔ Oklab Conversions ──────────────────────────────────

class TestOklabConversions:

    def test_white_has_L_one(self, solid_white):
        """White in Oklab should have L≈1.0, a≈0.0, b≈0.0."""
        from src.color.oklab import srgb_to_oklab
        lab = srgb_to_oklab(solid_white)
        np.testing.assert_allclose(lab[0, 0, 0], 1.0, atol=0.01)  # L
        np.testing.assert_allclose(lab[0, 0, 1], 0.0, atol=0.01)  # a
        np.testing.assert_allclose(lab[0, 0, 2], 0.0, atol=0.01)  # b

    def test_black_has_L_zero(self, solid_black):
        """Black in Oklab should have L≈0.0, a≈0.0, b≈0.0."""
        from src.color.oklab import srgb_to_oklab
        lab = srgb_to_oklab(solid_black)
        np.testing.assert_allclose(lab[0, 0], [0.0, 0.0, 0.0], atol=0.01)

    def test_red_has_positive_a(self, solid_red):
        """Red in Oklab should have positive 'a' (red-green axis)."""
        from src.color.oklab import srgb_to_oklab
        lab = srgb_to_oklab(solid_red)
        assert lab[0, 0, 1] > 0.05, "Red should have positive 'a' in Oklab"

    def test_oklab_roundtrip_accuracy(self, gradient_image):
        """sRGB → Oklab → sRGB roundtrip should be accurate to <0.001."""
        from src.color.oklab import srgb_to_oklab, oklab_to_srgb
        oklab = srgb_to_oklab(gradient_image)
        roundtrip = oklab_to_srgb(oklab)
        max_error = np.abs(roundtrip - gradient_image).max()
        assert max_error < 0.001, f"Roundtrip error too high: {max_error}"

    def test_oklab_output_shape(self, gradient_image):
        """Output shape must match input."""
        from src.color.oklab import srgb_to_oklab
        lab = srgb_to_oklab(gradient_image)
        assert lab.shape == gradient_image.shape

    def test_oklab_output_is_float32(self, gradient_image):
        """Output must be float32."""
        from src.color.oklab import srgb_to_oklab
        lab = srgb_to_oklab(gradient_image)
        assert lab.dtype == np.float32


# ── Oklab Utility Functions ────────────────────────────────────

class TestOklabUtilities:

    def test_oklab_saturation(self, solid_red):
        """Oklab chroma (saturation) should be computable as sqrt(a² + b²)."""
        from src.color.oklab import srgb_to_oklab, oklab_chroma
        lab = srgb_to_oklab(solid_red)
        chroma = oklab_chroma(lab)
        assert chroma.shape == (10, 10)
        assert chroma[0, 0] > 0.1, "Red should have significant chroma"

    def test_oklab_hue(self, solid_red):
        """Oklab hue angle should be computable via atan2(b, a)."""
        from src.color.oklab import srgb_to_oklab, oklab_hue
        lab = srgb_to_oklab(solid_red)
        hue = oklab_hue(lab)
        assert hue.shape == (10, 10)
        # Red hue in Oklab is approximately 29° (0.51 rad)
        assert 0.2 < hue[0, 0] < 0.8, f"Red hue unexpected: {hue[0, 0]}"
