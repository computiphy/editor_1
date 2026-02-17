import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple

class WatermarkEngine:
    def __init__(self, watermark_data: Optional[np.ndarray] = None, opacity: float = 0.5):
        self.watermark = watermark_data
        self.opacity = opacity

    def apply(self, image: np.ndarray, position: str = "bottom-right") -> np.ndarray:
        """
        Overlays a watermark on the image.
        Positions: "bottom-right", "bottom-left", "top-right", "top-left", "auto"
        """
        if self.watermark is None:
            return image

        h, w = image.shape[:2]
        wh, ww = self.watermark.shape[:2]

        if position == "auto":
            position = self._find_best_corner(image)

        # Map position to coordinates
        p_map = {
            "top-left": (0, 0),
            "top-right": (0, w - ww),
            "bottom-left": (h - wh, 0),
            "bottom-right": (h - wh, w - ww)
        }
        y, x = p_map.get(position, (h - wh, w - ww))

        # Alpha blending
        overlay = self.watermark[:, :, :3]
        alpha = self.watermark[:, :, 3] / 255.0 * self.opacity
        alpha = np.expand_dims(alpha, axis=-1)

        result = image.copy()
        region = result[y:y+wh, x:x+ww]
        result[y:y+wh, x:x+ww] = (1.0 - alpha) * region + alpha * overlay

        return result.astype(np.uint8)

    def _find_best_corner(self, image: np.ndarray) -> str:
        """
        Finds the corner with the lowest variance (detail).
        """
        h, w = image.shape[:2]
        pad = 100 # check 100x100 corner areas
        
        corners = {
            "top-left": image[:pad, :pad],
            "top-right": image[:pad, -pad:],
            "bottom-left": image[-pad:, :pad],
            "bottom-right": image[-pad:, -pad:]
        }
        
        best_corner = "bottom-right"
        min_var = float('inf')
        
        for name, region in corners.items():
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            var = np.var(gray)
            if var < min_var:
                min_var = var
                best_corner = name
                
        return best_corner
