from typing import List, Dict, Any
from src.core.models import LayoutPage
from src.core.enums import LayoutAlgorithm

class LayoutEngine:
    def __init__(self, algorithm: LayoutAlgorithm = LayoutAlgorithm.FIXED_PARTITION):
        self.algorithm = algorithm

    def generate_page(self, aspect_ratios: List[float], page_number: int) -> LayoutPage:
        """
        Generates a page layout for a list of image aspect ratios.
        """
        cells = []
        n = len(aspect_ratios)
        
        if self.algorithm == LayoutAlgorithm.FIXED_PARTITION:
            cells = self._row_layout(aspect_ratios)
        else:
            # Default to single image if unknown
            cells = [{"x": 0, "y": 0, "w": 1, "h": 1}] if n > 0 else []

        return LayoutPage(
            page_number=page_number,
            algorithm=self.algorithm.value,
            cells=cells
        )

    def _row_layout(self, ratios: List[float]) -> List[Dict[str, float]]:
        """
        Simple normalized row layout. Fits all images into a single horizontal row.
        Each image w is proportional to its aspect ratio.
        """
        total_ratio = sum(ratios)
        cells = []
        current_x = 0
        
        for r in ratios:
            w = r / total_ratio
            cells.append({
                "x": current_x,
                "y": 0,
                "w": w,
                "h": 1
            })
            current_x += w
            
        return cells
