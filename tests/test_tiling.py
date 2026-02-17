import pytest
import numpy as np
from src.utils.tiling import ImageTiler

def test_tiler_splits_and_merges():
    """
    Rationale: Validates that an image can be split into tiles and reconstructed perfectly.
    Details: This is critical for 4GB VRAM hardware, where processing a full-resolution 
    image at once would cause an OOM error.
    """
    # 512x512 image
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    tiler = ImageTiler(tile_size=256, overlap=32)
    
    tiles, metadata = tiler.split(img)
    
    # Check number of tiles: 512 + overlap logic
    # (512-32)/(256-32) = 480/224 = 2.14 -> 3 tiles per axis -> 9 tiles total
    assert len(tiles) > 1
    
    reconstructed = tiler.merge(tiles, metadata)
    assert reconstructed.shape == img.shape
    # Note: Overlap merging might have slight feathering, but matching shape is baseline
    assert np.array_equal(img, reconstructed) # Precise check for no-op tiles
