import pytest
import numpy as np
from PIL import Image
from src.watermark.engine import WatermarkEngine

@pytest.fixture
def dummy_image():
    return np.zeros((1000, 1000, 3), dtype=np.uint8)

@pytest.fixture
def dummy_watermark():
    # 100x100 white square with transparency
    data = np.ones((100, 100, 4), dtype=np.uint8) * 255
    data[:, :, 3] = 128 # 50% alpha
    return data

def test_watermark_placement_bottom_right(dummy_image, dummy_watermark):
    """
    Rationale: Verifies that the watermark can be placed in the bottom-right corner.
    Details: Basic placement logic for brand consistency.
    """
    engine = WatermarkEngine(watermark_data=dummy_watermark)
    
    # Place manually or auto
    result = engine.apply(dummy_image, position="bottom-right")
    
    assert result.shape == dummy_image.shape
    # Bottom right pixel should be changed (if not black already)
    # Since dummy_image is black and watermark is white, checking the pixel is easy
    assert np.mean(result[950:, 950:, :]) > 0

def test_watermark_auto_placement(dummy_image, dummy_watermark):
    """
    Rationale: Verifies the auto-placement logic based on image detail.
    Details: Ensures the system picks a low-detail region to avoid distracting from the subject.
    """
    # Create an image with more detail at the top than the bottom
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    img[:500, :, :] = np.random.randint(0, 255, (500, 1000, 3)) # Busy top
    # Bottom remains black (low detail)
    
    engine = WatermarkEngine(watermark_data=dummy_watermark)
    result = engine.apply(img, position="auto")
    
    # Should pick a bottom corner. Let's check if the top remains unchanged.
    assert np.array_equal(result[:100, :100, :], img[:100, :100, :])
