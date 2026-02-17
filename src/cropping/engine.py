import numpy as np
from typing import List, Tuple, Optional
from src.core.enums import CropRatio

class CroppingEngine:
    def __init__(self):
        pass

    def get_crop(self, image: np.ndarray, ratio: CropRatio = CropRatio.SQUARE) -> np.ndarray:
        """
        Calculates and returns a crop of the image matching the target ratio.
        Centers on detected subjects.
        """
        h, w = image.shape[:2]
        subjects = self._detect_subjects(image)
        
        # Parse ratio string e.g. "9:16"
        r_w, r_h = map(int, ratio.value.split(':'))
        target_ratio = r_w / r_h

        # Determine dimensions of the crop window
        if (w / h) > target_ratio:
            # Source is wider than target: fit to height
            new_h = h
            new_w = int(h * target_ratio)
        else:
            # Source is taller than target: fit to width
            new_w = w
            new_h = int(w / target_ratio)

        # Center logic
        if not subjects:
            # Center of the image
            cx, cy = w // 2, h // 2
        else:
            # Average center of all detected subjects
            # subjects is list of (x1, y1, x2, y2)
            centers = [((s[0]+s[2])//2, (s[1]+s[3])//2) for s in subjects]
            cx = sum(c[0] for c in centers) // len(centers)
            cy = sum(c[1] for c in centers) // len(centers)

        # Calculate bounds
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        
        # Clamp to image boundaries
        if x1 + new_w > w: x1 = w - new_w
        if y1 + new_h > h: y1 = h - new_h
        
        # If still out of bounds (can happen if target is larger than source, 
        # but here we fitted to min side), clamp again.
        x2 = min(x1 + new_w, w)
        y2 = min(y1 + new_h, h)

        return image[y1:y2, x1:x2]

    def _detect_subjects(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Mock RT-DETR detection. Returns bounding boxes.
        """
        return []
