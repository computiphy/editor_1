import pytest
import numpy as np
from src.culling.quality_assessor import QualityAssessor

@pytest.fixture
def dummy_image():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

def test_quality_assessor_returns_scores(dummy_image):
    """
    Rationale: Validates the interface for advanced perceptual quality metrics.
    Details: This test ensures that the `QualityAssessor` can return BRISQUE and NIQE scores, 
    which are used for Phase 2 culling and AI auto-routing.
    """
    # This might fail if models are not found, we'll handle mock if needed
    assessor = QualityAssessor()
    brisque, niqe = assessor.calculate_scores(dummy_image)
    
    assert isinstance(brisque, float)
    assert isinstance(niqe, float)
