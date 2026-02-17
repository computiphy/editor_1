import pytest
import numpy as np
from src.culling.blur_detector import BlurDetector

def test_blur_detector_low_variance_is_blurry():
    """
    Rationale: Flags images with low Laplacian variance as blurry.
    Details: This is a core part of the Stage 1 culling engine, used to filter out out-of-focus shots.
    """
    # Create a blurry image (constant)
    blurry_img = np.zeros((100, 100), dtype=np.uint8)
    detector = BlurDetector(threshold=100.0)
    
    score = detector.calculate_variance(blurry_img)
    assert score < 100.0
    assert detector.is_blurry(blurry_img) == True

def test_blur_detector_high_variance_is_sharp():
    """
    Rationale: Correctly identifies sharp images.
    Details: Ensures that sharp images pass the quality threshold, preventing excessive rejection.
    """
    # Create a sharp image (checkerboard)
    sharp_img = np.array([[0, 255] * 50, [255, 0] * 50] * 50, dtype=np.uint8)
    detector = BlurDetector(threshold=100.0)
    
    score = detector.calculate_variance(sharp_img)
    assert score > 100.0
    assert detector.is_blurry(sharp_img) == False
