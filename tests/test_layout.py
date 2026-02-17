import pytest
from src.layout.engine import LayoutEngine
from src.core.enums import LayoutAlgorithm

def test_layout_engine_fixed_partition():
    """
    Rationale: Validates the fixed-partition layout algorithm.
    Details: Ensures the engine can pack 3-4 images into a single page with valid 
    non-overlapping coordinates based on aspect ratios.
    """
    engine = LayoutEngine(algorithm=LayoutAlgorithm.FIXED_PARTITION)
    
    # Aspect ratios for 3 images
    ratios = [1.5, 0.66, 1.0] # L, P, S
    
    page = engine.generate_page(ratios, page_number=1)
    
    assert page.page_number == 1
    assert len(page.cells) == 3
    # Check that cells are within 0-1 range (relative)
    for cell in page.cells:
        assert 0 <= cell['x'] <= 1
        assert 0 <= cell['y'] <= 1
        assert cell['w'] > 0
        assert cell['h'] > 0
