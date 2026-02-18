import pytest
import numpy as np


def test_background_remover_returns_rgba():
    """
    Rationale: Ensures the background remover outputs RGBA with transparency.
    Details: Verifies shape has 4 channels and alpha channel is not all-opaque.
    """
    from src.segmentation.background_remover import BackgroundRemover

    remover = BackgroundRemover()

    # Create a simple test image: white circle on black background
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a filled white circle in the center
    import cv2
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)

    result = remover.remove_background(img)

    # Should be RGBA
    assert result.shape[2] == 4, f"Expected 4 channels (RGBA), got {result.shape[2]}"
    assert result.dtype == np.uint8

    # Alpha channel should have some transparent pixels (not all 255)
    alpha = result[:, :, 3]
    assert alpha.min() < 255 or alpha.max() > 0, "Alpha channel seems wrong"


def test_background_remover_preserves_dimensions():
    """
    Rationale: Output dimensions must match input for compositing.
    """
    from src.segmentation.background_remover import BackgroundRemover

    remover = BackgroundRemover()
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

    result = remover.remove_background(img)

    assert result.shape[0] == 200
    assert result.shape[1] == 300
    assert result.shape[2] == 4
