"""
Background Selector
====================
Selects the best background image for an album page based on
dominant color matching using LAB ΔE distance.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from src.layout.color_analyzer import (
    extract_dominant_color,
    compute_average_color,
    delta_e,
)


class BackgroundSelector:
    """
    Selects background images by matching page color palette to
    candidate backgrounds using ΔE scoring in LAB space.
    """

    def __init__(self, background_dir: Path, strategy: str = "dominant"):
        """
        Args:
            background_dir: Directory containing background JPEG files.
            strategy: "dominant" (match strongest color) or "average" (mean LAB).
        """
        self.background_dir = background_dir
        self.strategy = strategy
        self._candidates = []
        self._candidate_colors = {}

        # Pre-compute colors for all candidate backgrounds
        if background_dir.exists():
            for f in sorted(background_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self._candidates.append(f)

            for bg_path in self._candidates:
                try:
                    color = extract_dominant_color(bg_path)
                    self._candidate_colors[bg_path] = color
                except Exception as e:
                    print(f"Warning: Could not analyze background {bg_path}: {e}")

    def select(self, image_paths: List[Path]) -> Optional[Path]:
        """
        Select the best background for a set of images.

        Args:
            image_paths: Paths to the images being placed on this page.

        Returns:
            Path to the best-matching background, or None if no candidates.
        """
        if not self._candidates or not self._candidate_colors:
            return None

        # Compute page color
        if self.strategy == "average":
            page_color = compute_average_color(image_paths)
        else:
            # "dominant" strategy: use dominant color of the first image
            # (which is the hero/most important image on the page)
            page_color = extract_dominant_color(image_paths[0])

        # Score all candidates by ΔE distance
        best_path = None
        best_score = float("inf")

        for bg_path, bg_color in self._candidate_colors.items():
            score = delta_e(page_color, bg_color)
            if score < best_score:
                best_score = score
                best_path = bg_path

        return best_path

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)
