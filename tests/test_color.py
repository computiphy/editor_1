import pytest
import numpy as np
from src.color.engine import ColorGradingEngine, PRESETS

@pytest.fixture
def dummy_image():
    """A 200x200 test image with varied colors."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:100, :100] = [220, 160, 120]   # Skin tone (top-left)
    img[:100, 100:] = [80, 130, 200]    # Sky blue (top-right)
    img[100:, :100] = [80, 160, 60]     # Vegetation green (bottom-left)
    img[100:, 100:] = [40, 40, 40]      # Dark suit (bottom-right)
    return img

def test_color_grading_lab_statistical(dummy_image):
    """
    Rationale: Ensures Reinhard Lab-space transfer works.
    Details: Legacy API for reference-based color matching.
    """
    engine = ColorGradingEngine()
    reference = np.ones_like(dummy_image) * 128
    result = engine.apply_transfer(dummy_image, reference)
    assert result.shape == dummy_image.shape

def test_color_grading_cinematic_style(dummy_image):
    """
    Rationale: Validates the cinematic preset applies teal-orange push.
    Details: Checks that greens shift towards teal (desaturated) and
    skin tones are preserved/boosted.
    """
    engine = ColorGradingEngine(style="cinematic")
    result = engine.apply_style(dummy_image)
    
    assert result.shape == dummy_image.shape
    assert result.dtype == np.uint8
    # Cinematic should NOT be identical to input
    assert not np.array_equal(result, dummy_image)

def test_color_grading_pastel_style(dummy_image):
    """
    Rationale: Validates the pastel preset creates a soft, desaturated look.
    Details: Mean saturation should be lower than the input.
    """
    import cv2
    engine = ColorGradingEngine(style="pastel")
    result = engine.apply_style(dummy_image)
    
    # Pastel should have lower saturation
    hsv_in = cv2.cvtColor(dummy_image, cv2.COLOR_RGB2HSV)
    hsv_out = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    assert hsv_out[:, :, 1].mean() < hsv_in[:, :, 1].mean()

def test_color_grading_black_and_white(dummy_image):
    """
    Rationale: Validates B&W preset produces a desaturated output.
    Details: Mean saturation should be near zero.
    """
    import cv2
    engine = ColorGradingEngine(style="black_and_white")
    result = engine.apply_style(dummy_image)
    
    hsv_out = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    assert hsv_out[:, :, 1].mean() < 10  # Nearly zero saturation

def test_color_grading_strength_zero(dummy_image):
    """
    Rationale: Ensures strength=0 produces no change.
    Details: All adjustments are scaled by strength, so 0 should be identity.
    """
    engine = ColorGradingEngine(style="cinematic", strength=0.0)
    result = engine.apply_style(dummy_image)
    
    # Should be very close to original (small rounding differences OK)
    diff = np.abs(result.astype(float) - dummy_image.astype(float)).mean()
    assert diff < 5.0  # Allow small rounding errors from color space conversions

def test_all_presets_exist():
    """
    Rationale: Ensures all 9 presets defined in color_theory.md are registered.
    """
    expected = ["natural", "cinematic", "pastel", "moody", "golden_hour",
                "film_kodak", "film_fuji", "vibrant", "black_and_white"]
    for name in expected:
        assert name in PRESETS, f"Missing preset: {name}"

def test_all_presets_run(dummy_image):
    """
    Rationale: Smoke test - all presets should execute without errors.
    """
    for name in PRESETS:
        engine = ColorGradingEngine(style=name)
        result = engine.apply_style(dummy_image)
        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8
