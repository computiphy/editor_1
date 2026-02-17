import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from src.utils.image_io import load_image, save_image

def test_load_save_jpeg(tmp_path):
    """
    Rationale: Ensures basic image I/O works for standard formats.
    Details: This test verifies that the system can read JPEGs into NumPy arrays and write them 
    back as TIFFs, which is a basic requirement for the processing pipeline.
    """
    # Create test image
    img_path = tmp_path / "test.jpg"
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(data).save(img_path)
    
    # Load
    loaded = load_image(str(img_path))
    assert loaded.shape == (100, 100, 3)
    
    # Save as TIFF
    out_path = tmp_path / "test.tiff"
    save_image(loaded, str(out_path))
    assert out_path.exists()

def test_develop_raw_mock(mocker):
    """
    Rationale: Validates the RAW development interface without needing a heavy RAW file fixture.
    Details: This test mocks `rawpy` to ensure that the `develop_raw` function correctly calls 
    the postprocessing engine, maintaining a "RAW first" workflow.
    """
    # Mock rawpy since we don't have a real RAW file
    import rawpy
    mock_raw = mocker.patch("rawpy.imread")
    mock_context = mock_raw.return_value.__enter__.return_value
    mock_context.postprocess.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    from src.utils.image_io import develop_raw
    result = develop_raw("dummy.dng")
    assert result.shape == (100, 100, 3)
