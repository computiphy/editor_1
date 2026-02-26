"""
Tests for SOTA Engine P2 + P3 Features
========================================
P2: CLAHE in Oklab, Frequency Separation for Skin
P3: Halation (red-channel scattering)
"""

import pytest
import numpy as np


@pytest.fixture
def flat_image():
    """A flat, low-contrast image (simulates poorly lit reception hall)."""
    # Narrow dynamic range: all values between 0.3 and 0.5
    rng = np.random.RandomState(42)
    return (rng.rand(200, 200, 3).astype(np.float32) * 0.2 + 0.3)


@pytest.fixture
def skin_image():
    """An image with a uniform skin-tone region and some texture."""
    img = np.full((200, 200, 3), [220/255, 170/255, 140/255], dtype=np.float32)
    # Add high-frequency texture (simulates pores/wrinkles)
    rng = np.random.RandomState(42)
    texture = rng.rand(200, 200, 3).astype(np.float32) * 0.03
    img += texture
    return np.clip(img, 0.0, 1.0)


@pytest.fixture
def highlight_image():
    """An image with bright highlights for halation testing."""
    img = np.full((200, 200, 3), 0.3, dtype=np.float32)
    # Bright highlight spot in center
    img[80:120, 80:120] = 0.95
    return img


# ── P2: CLAHE in Oklab ────────────────────────────────────────

class TestCLAHEOklab:

    def test_clahe_increases_contrast(self, flat_image):
        """CLAHE should increase the dynamic range of a flat image."""
        from src.color.engine_v2 import apply_clahe_oklab
        result = apply_clahe_oklab(flat_image, clip_limit=2.0, grid_size=8)

        # Dynamic range should increase
        input_range = flat_image.max() - flat_image.min()
        output_range = result.max() - result.min()
        assert output_range > input_range, (
            f"CLAHE should increase range: {input_range:.3f} → {output_range:.3f}"
        )

    def test_clahe_preserves_color(self, flat_image):
        """CLAHE should modify lightness, not hue/saturation significantly."""
        import cv2
        from src.color.engine_v2 import apply_clahe_oklab

        result = apply_clahe_oklab(flat_image, clip_limit=2.0, grid_size=8)

        # Convert to HSV and compare hue channels
        hsv_in = cv2.cvtColor((flat_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_out = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

        hue_diff = np.abs(hsv_out[:, :, 0].astype(float) - hsv_in[:, :, 0].astype(float)).mean()
        assert hue_diff < 10.0, f"CLAHE should preserve hue, diff={hue_diff:.1f}"

    def test_clahe_output_shape_and_dtype(self, flat_image):
        """Output should preserve shape and be float32."""
        from src.color.engine_v2 import apply_clahe_oklab
        result = apply_clahe_oklab(flat_image)
        assert result.shape == flat_image.shape
        assert result.dtype == np.float32

    def test_clahe_handles_already_contrasty(self):
        """CLAHE should not blow out an already well-exposed image."""
        from src.color.engine_v2 import apply_clahe_oklab
        # Full dynamic range image
        good = np.linspace(0, 1, 200*200*3).reshape(200, 200, 3).astype(np.float32)
        result = apply_clahe_oklab(good, clip_limit=2.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ── P2: Frequency Separation for Skin ─────────────────────────

class TestFrequencySeparation:

    def test_separation_preserves_texture(self, skin_image):
        """
        After frequency separation and low-freq color correction,
        the high-frequency texture (pores) must be preserved.
        """
        from src.color.engine_v2 import frequency_separate, frequency_merge

        low, high = frequency_separate(skin_image, blur_radius=5)

        # Modify only the low-frequency (color correction simulation)
        low_modified = low * 0.95  # Slight darkening

        # Recombine
        result = frequency_merge(low_modified, high)

        # The high-frequency detail should survive
        # Compare standard deviation of result (texture) vs original
        original_std = skin_image.std()
        result_std = result.std()
        ratio = result_std / max(original_std, 1e-6)
        assert ratio > 0.7, f"Texture lost: std ratio = {ratio:.3f}"

    def test_low_freq_is_smooth(self, skin_image):
        """The low-frequency component should be smoother than the original."""
        from src.color.engine_v2 import frequency_separate

        low, high = frequency_separate(skin_image, blur_radius=5)
        assert low.std() < skin_image.std(), "Low-freq should be smoother"

    def test_roundtrip_identity(self, skin_image):
        """Separating and merging without changes should return the original."""
        from src.color.engine_v2 import frequency_separate, frequency_merge

        low, high = frequency_separate(skin_image, blur_radius=5)
        result = frequency_merge(low, high)
        max_error = np.abs(result - skin_image).max()
        assert max_error < 0.01, f"Roundtrip error: {max_error:.4f}"

    def test_output_shapes(self, skin_image):
        """Both low and high frequency should match input shape."""
        from src.color.engine_v2 import frequency_separate
        low, high = frequency_separate(skin_image, blur_radius=5)
        assert low.shape == skin_image.shape
        assert high.shape == skin_image.shape


# ── P3: Halation ──────────────────────────────────────────────

class TestHalation:

    def test_halation_adds_red_glow(self, highlight_image):
        """Halation should add red glow around bright highlights."""
        from src.color.engine_v2 import apply_halation

        result = apply_halation(highlight_image, intensity=0.5, radius=15)

        # Pixels near (but not in) the highlight should have more red than original
        # Check the border area around the highlight (e.g., row 75)
        border_red_before = highlight_image[75, 80:120, 0].mean()
        border_red_after = result[75, 80:120, 0].mean()
        assert border_red_after > border_red_before, (
            f"Halation should add red glow: {border_red_before:.3f} → {border_red_after:.3f}"
        )

    def test_halation_only_affects_red(self, highlight_image):
        """Halation should primarily affect the red channel."""
        from src.color.engine_v2 import apply_halation

        result = apply_halation(highlight_image, intensity=0.5, radius=15)
        diff = result - highlight_image

        # Red channel should change more than blue
        red_diff = np.abs(diff[:, :, 0]).mean()
        blue_diff = np.abs(diff[:, :, 2]).mean()
        assert red_diff > blue_diff * 2, (
            f"Red should change most: R={red_diff:.4f}, B={blue_diff:.4f}"
        )

    def test_halation_zero_intensity(self, highlight_image):
        """Zero intensity should produce no change."""
        from src.color.engine_v2 import apply_halation
        result = apply_halation(highlight_image, intensity=0.0)
        np.testing.assert_allclose(result, highlight_image, atol=1e-6)

    def test_halation_output_range(self, highlight_image):
        """Output should stay in [0, 1] range."""
        from src.color.engine_v2 import apply_halation
        result = apply_halation(highlight_image, intensity=0.8, radius=20)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
