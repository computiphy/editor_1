import numpy as np
from PIL import Image
import rawpy
from typing import Union

def load_image(path: str) -> np.ndarray:
    """Load an image file (JPEG, PNG, TIFF) into a NumPy array."""
    with Image.open(path) as img:
        return np.array(img)

def save_image(data: np.ndarray, path: str):
    """Save a NumPy array as an image file."""
    img = Image.fromarray(data)
    img.save(path)

def develop_raw(path: str, use_camera_wb: bool = True, bright: float = 1.0) -> np.ndarray:
    """Develop a RAW file into a NumPy array using rawpy."""
    with rawpy.imread(path) as raw:
        return raw.postprocess(use_camera_wb=use_camera_wb, bright=bright)
