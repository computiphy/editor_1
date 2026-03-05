import numpy as np
from PIL import Image, ImageOps
import rawpy
from typing import Union

def load_image(path: str) -> np.ndarray:
    """
    Load an image file (JPEG, PNG, TIFF) OR a RAW file (DNG, ARW, CR2, etc.)
    into a NumPy array. Applies EXIF orientation for standard formats
    and high-quality development for RAW formats.
    """
    from pathlib import Path
    ext = Path(path).suffix.lower()
    
    # Common RAW formats handled by rawpy
    raw_exts = {'.dng', '.arw', '.cr2', '.nef', '.orf', '.raf', '.sr2'}
    
    if ext in raw_exts:
        try:
            return develop_raw(path)
        except Exception as e:
            # If rawpy fails (e.g., file corrupted), try PIL as a fallback
            pass

    with Image.open(path) as img:
        # Apply EXIF orientation tag to actual pixel data for standard formats.
        img = ImageOps.exif_transpose(img)
        return np.array(img)

def save_image(data: np.ndarray, path: str, output_format: str = "jpeg"):
    """
    Save a NumPy array as an image file.
    Supports formats: 'jpeg', 'png', 'tiff', 'original'.
    If 'original', it tries to use the extension from 'path', 
    but falls back to JPEG for unsupported RAW extensions (DNG, CR2, etc.).
    """
    from pathlib import Path
    
    # 1. Resolve Extension
    p = Path(path)
    ext = p.suffix.lower()
    
    # Common RAW formats we cannot write
    raw_exts = {'.dng', '.arw', '.cr2', '.nef', '.orf', '.raf', '.sr2'}
    
    if output_format == "original":
        # Keep original extension if it's a standard format, otherwise JPEG
        if ext in raw_exts or not ext:
            final_path = p.with_suffix(".jpg")
        else:
            final_path = p
    else:
        # Use simple mapping for standard formats
        format_map = {"jpeg": ".jpg", "png": ".png", "tiff": ".tif"}
        target_ext = format_map.get(output_format.lower(), ".jpg")
        final_path = p.with_suffix(target_ext)

    # 2. Save using Pillow
    img = Image.fromarray(data)
    
    # Apply format-specific optimizations
    save_args = {}
    if final_path.suffix in {".jpg", ".jpeg"}:
        save_args = {"quality": 95, "subsampling": 0}
    elif final_path.suffix == ".png":
        save_args = {"optimize": True}
    elif final_path.suffix in {".tif", ".tiff"}:
        save_args = {"compression": "tiff_lzw"}

    final_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(final_path), **save_args)
    return final_path

def develop_raw(path: str, use_camera_wb: bool = True, bright: float = 1.0) -> np.ndarray:
    """Develop a RAW file into a NumPy array using rawpy."""
    with rawpy.imread(path) as raw:
        return raw.postprocess(use_camera_wb=use_camera_wb, bright=bright)
