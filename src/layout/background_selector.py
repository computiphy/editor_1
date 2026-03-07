"""
Background Selector
====================
Selects a background image for each album page based on a configurable
selection strategy.

Strategies
----------
dominant    : Best LAB ΔE match using the dominant colour of each page (default).
average     : Best LAB ΔE match using the mean colour of all images on the page.
random      : Pick a random background each page (different seed per call).
round_robin : Cycle through the pool sequentially (1, 2, 3, … 1, 2, 3, …).
shuffle     : Shuffle the pool once at startup, then cycle through in that order.
fixed       : Always use the first background in the pool (useful for testing).
"""

import random as _random
from pathlib import Path
from typing import List, Optional

from src.layout.color_analyzer import (
    extract_dominant_color,
    compute_average_color,
    delta_e,
)

# All strategies that require LAB colour analysis
_COLOR_MATCH_STRATEGIES = {"dominant", "average"}


class BackgroundSelector:
    """
    Selects background images for album pages.

    Parameters
    ----------
    background_dir : Path
        Directory that contains candidate background image files.
    strategy : str
        One of: "dominant", "average", "random", "round_robin", "shuffle", "fixed".
    seed : int | None
        Optional random seed for reproducible ``random`` / ``shuffle`` results.
    """

    VALID_STRATEGIES = {
        "dominant", "average", "random", "round_robin", "shuffle", "fixed"
    }

    def __init__(
        self,
        background_dir: Path,
        strategy: str = "dominant",
        seed: Optional[int] = None,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown background strategy '{strategy}'. "
                f"Valid options: {sorted(self.VALID_STRATEGIES)}"
            )

        self.background_dir = background_dir
        self.strategy = strategy
        self._rng = _random.Random(seed)

        # Discover all background files
        self._candidates: List[Path] = []
        if background_dir.exists():
            for f in sorted(background_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self._candidates.append(f)

        # Pre-compute LAB colours only when we need them
        self._candidate_colors: dict = {}
        if self._candidates and strategy in _COLOR_MATCH_STRATEGIES:
            for bg_path in self._candidates:
                try:
                    self._candidate_colors[bg_path] = extract_dominant_color(bg_path)
                except Exception as e:
                    print(f"Warning: Could not analyse background {bg_path}: {e}")

        # Build the cycling pool for round_robin / shuffle
        self._pool: List[Path] = list(self._candidates)
        if strategy == "shuffle" and self._pool:
            self._rng.shuffle(self._pool)
        self._pool_index: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def select(self, image_paths: List[Path]) -> Optional[Path]:
        """
        Select a background for the given page images.

        Parameters
        ----------
        image_paths : list of Path
            The images being placed on this page (used for colour matching).

        Returns
        -------
        Path | None
            Path to the chosen background, or ``None`` if the pool is empty.
        """
        if not self._candidates:
            return None

        strategy = self.strategy

        # ── Colour-match strategies ───────────────────────────────────────────
        if strategy in _COLOR_MATCH_STRATEGIES:
            if not self._candidate_colors:
                # Fallback: nothing was colour-analysed successfully
                return self._candidates[0]

            if strategy == "average":
                page_color = compute_average_color(image_paths)
            else:  # "dominant"
                page_color = extract_dominant_color(image_paths[0])

            best_path = min(
                self._candidate_colors,
                key=lambda bg: delta_e(page_color, self._candidate_colors[bg]),
            )
            return best_path

        # ── Stateless random ─────────────────────────────────────────────────
        if strategy == "random":
            return self._rng.choice(self._candidates)

        # ── Fixed (always first) ─────────────────────────────────────────────
        if strategy == "fixed":
            return self._candidates[0]

        # ── Cycling strategies (round_robin, shuffle) ─────────────────────────
        if strategy in ("round_robin", "shuffle"):
            bg = self._pool[self._pool_index % len(self._pool)]
            self._pool_index += 1
            return bg

        # Shouldn't reach here, but be safe
        return self._candidates[0]

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def candidate_count(self) -> int:
        """Number of background files in the pool."""
        return len(self._candidates)

    def __repr__(self) -> str:
        return (
            f"BackgroundSelector(strategy={self.strategy!r}, "
            f"candidates={self.candidate_count})"
        )
