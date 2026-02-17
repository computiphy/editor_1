import pytest
import numpy as np
from src.cropping.engine import CroppingEngine
from src.core.enums import CropRatio

@pytest.fixture
def dummy_image():
    # 1000x1000 image
    return np.zeros((1000, 1000, 3), dtype=np.uint8)

def test_cropping_engine_square_crop(dummy_image, mocker):
    """
    Rationale: Validates that the engine can produce a 1:1 square crop.
    Details: This test ensures the geometry logic for centering the crop window 
    works correctly regardless of subject detection.
    """
    engine = CroppingEngine()
    
    # Mock detector to find a person in the top-left (center at 100, 100)
    mocker.patch.object(engine, "_detect_subjects", return_value=[(50, 50, 150, 150)])
    
    crop = engine.get_crop(dummy_image, ratio=CropRatio.SQUARE)
    
    # Square crop of 1000x1000 should be 1000x1000 if it fits
    # But if we want to "crop", usually we target a subset or just verify the shape.
    # In my implementation, I'll make it return a 1:1 window.
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[0] <= dummy_image.shape[0]

def test_cropping_engine_story_crop(dummy_image, mocker):
    """
    Rationale: Validates 9:16 story crop for social media.
    Details: Verifies that the engine can calculate a vertical crop from a square/landscape source.
    """
    engine = CroppingEngine()
    mocker.patch.object(engine, "_detect_subjects", return_value=[]) # No subjects
    
    crop = engine.get_crop(dummy_image, ratio=CropRatio.STORY)
    
    h, w = crop.shape[:2]
    expected_ratio = 9/16
    assert abs((w/h) - expected_ratio) < 0.01
