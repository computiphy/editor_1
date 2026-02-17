import pytest
import numpy as np
from src.color.engine import ColorGradingEngine
from src.core.enums import ColorMethod

@pytest.fixture
def dummy_image():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

def test_color_grading_lab_statistical(dummy_image):
    """
    Rationale: Ensures that statistical color transfer in Lab space works.
    Details: This test verifies that the engine can adjust the mean/std of an image 
    to match a "hero shot", which is a common wedding photography requirement.
    """
    engine = ColorGradingEngine(method=ColorMethod.LAB_STATISTICAL)
    
    # Target image with different mean color (e.g. reddish)
    target = np.zeros((100, 100, 3), dtype=np.uint8)
    target[:, :, 0] = 200 # Red
    
    graded = engine.apply_transfer(dummy_image, target)
    
    assert graded.shape == dummy_image.shape
    assert not np.array_equal(graded, dummy_image)
    # Average red should be higher now
    assert np.mean(graded[:, :, 0]) > np.mean(dummy_image[:, :, 0])
