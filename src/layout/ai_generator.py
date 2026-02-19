"""
AI Layout Generator (Phase 3 — Rule-Based)
============================================
Generates visually balanced album page layouts using mathematical
optimization. No ML model required — uses compositional principles:

  1. Golden Ratio Partitioning (φ = 1.618)
  2. Visual Weight Balancing (hero images get more area)
  3. Aspect Ratio Matching (tall images → tall cells)
  4. Style-aware modifiers (elegant, dynamic, minimal, classic, magazine)
  5. Seed-based reproducibility

Outputs the same normalized cell format as TemplateRegistry,
so the renderer works identically regardless of generator source.
"""

import math
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

# Golden Ratio
PHI = 1.6180339887


class LayoutStyle(Enum):
    ELEGANT = "elegant"       # Generous whitespace, golden ratio, max emphasis on hero
    MINIMAL = "minimal"       # Clean equal grid, balanced
    DYNAMIC = "dynamic"       # Varied sizes, asymmetric, energetic
    CLASSIC = "classic"       # Traditional balanced album layouts
    MAGAZINE = "magazine"     # Mixed hero + supporting shots, editorial feel


@dataclass
class GeneratedCell:
    """A generated cell in normalized coordinates (0.0–1.0)."""
    x: float
    y: float
    w: float
    h: float
    importance: float = 1.0   # Visual weight score (higher = more prominent)


@dataclass
class GeneratedLayout:
    """A complete generated layout with metadata."""
    name: str
    cells: List[GeneratedCell]
    style: str
    score: float              # Quality score (higher = better composition)
    seed: int


class AILayoutGenerator:
    """
    Rule-based layout generator using compositional mathematics.
    Generates multiple layout variants and scores them.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

    def generate(self, image_count: int,
                 aspect_ratios: Optional[List[float]] = None,
                 page_aspect: float = 1.5,
                 style: str = "classic",
                 num_variants: int = 3) -> GeneratedLayout:
        """
        Generate the best layout for a set of images.

        Args:
            image_count: Number of images to place.
            aspect_ratios: Aspect ratios (w/h) of each image. None = assume 1.5.
            page_aspect: Page width/height ratio.
            style: Layout style (elegant, minimal, dynamic, classic, magazine).
            num_variants: Number of variants to generate and score.

        Returns:
            The highest-scoring GeneratedLayout.
        """
        if aspect_ratios is None:
            aspect_ratios = [1.5] * image_count

        # Ensure we have the right number of ratios
        while len(aspect_ratios) < image_count:
            aspect_ratios.append(1.5)
        aspect_ratios = aspect_ratios[:image_count]

        # Generate variants with different seeds
        variants = []
        for i in range(num_variants):
            variant_seed = self.seed + i * 97  # Deterministic variant seeds
            rng = random.Random(variant_seed)
            layout = self._generate_variant(
                image_count, aspect_ratios, page_aspect, style, rng, variant_seed
            )
            variants.append(layout)

        # Return highest-scoring variant
        variants.sort(key=lambda v: v.score, reverse=True)
        return variants[0]

    def generate_all_variants(self, image_count: int,
                               aspect_ratios: Optional[List[float]] = None,
                               page_aspect: float = 1.5,
                               style: str = "classic",
                               num_variants: int = 5) -> List[GeneratedLayout]:
        """Generate and return ALL variants (sorted by score descending)."""
        if aspect_ratios is None:
            aspect_ratios = [1.5] * image_count
        while len(aspect_ratios) < image_count:
            aspect_ratios.append(1.5)
        aspect_ratios = aspect_ratios[:image_count]

        variants = []
        for i in range(num_variants):
            variant_seed = self.seed + i * 97
            rng = random.Random(variant_seed)
            layout = self._generate_variant(
                image_count, aspect_ratios, page_aspect, style, rng, variant_seed
            )
            variants.append(layout)

        variants.sort(key=lambda v: v.score, reverse=True)
        return variants

    def _generate_variant(self, count: int, ratios: List[float],
                           page_aspect: float, style: str,
                           rng: random.Random, seed: int) -> GeneratedLayout:
        """Generate a single layout variant."""
        style_enum = LayoutStyle(style) if style in [s.value for s in LayoutStyle] else LayoutStyle.CLASSIC

        if count == 1:
            cells = self._layout_single(style_enum, rng)
        elif count == 2:
            cells = self._layout_duo(ratios, style_enum, rng)
        elif count == 3:
            cells = self._layout_trio(ratios, style_enum, rng)
        elif count == 4:
            cells = self._layout_quad(ratios, style_enum, rng)
        elif count == 5:
            cells = self._layout_quint(ratios, style_enum, rng)
        elif count >= 6:
            cells = self._layout_grid(count, ratios, style_enum, rng)
        else:
            cells = []

        # Apply style modifiers
        cells = self._apply_style_modifiers(cells, style_enum, rng)

        # Score the layout
        score = self._score_layout(cells, ratios, page_aspect)

        name = f"ai_{style}_{count}img_s{seed}"
        return GeneratedLayout(
            name=name, cells=cells, style=style,
            score=score, seed=seed,
        )

    # ── Single Image Layouts ───────────────────────────────────

    def _layout_single(self, style: LayoutStyle, rng: random.Random) -> List[GeneratedCell]:
        """Single image — full bleed or centered with padding."""
        if style == LayoutStyle.ELEGANT:
            # Centered with generous golden-ratio padding
            pad = 1.0 - (1.0 / PHI)
            half_pad = pad / 2
            return [GeneratedCell(half_pad, half_pad, 1.0 - pad, 1.0 - pad, importance=1.0)]
        elif style == LayoutStyle.MINIMAL:
            # Centered with clean 10% padding
            return [GeneratedCell(0.10, 0.10, 0.80, 0.80, importance=1.0)]
        else:
            # Full bleed
            return [GeneratedCell(0.0, 0.0, 1.0, 1.0, importance=1.0)]

    # ── Duo Layouts ────────────────────────────────────────────

    def _layout_duo(self, ratios: List[float], style: LayoutStyle,
                     rng: random.Random) -> List[GeneratedCell]:
        """Two images — golden ratio split."""
        # Decide split direction based on ratios
        avg_ratio = sum(ratios) / len(ratios)
        horizontal = avg_ratio >= 1.0  # Landscape → split horizontally

        if style == LayoutStyle.DYNAMIC:
            # Asymmetric golden split
            split = 1.0 / PHI  # ≈ 0.618
        elif style == LayoutStyle.ELEGANT:
            split = 1.0 / PHI
        else:
            # Equal split
            split = 0.5

        # Randomly mirror the split for variety
        if rng.random() > 0.5:
            split = 1.0 - split

        if horizontal:
            return [
                GeneratedCell(0.0, 0.0, split, 1.0, importance=1.2 if split > 0.5 else 0.8),
                GeneratedCell(split, 0.0, 1.0 - split, 1.0, importance=0.8 if split > 0.5 else 1.2),
            ]
        else:
            return [
                GeneratedCell(0.0, 0.0, 1.0, split, importance=1.2 if split > 0.5 else 0.8),
                GeneratedCell(0.0, split, 1.0, 1.0 - split, importance=0.8 if split > 0.5 else 1.2),
            ]

    # ── Trio Layouts ───────────────────────────────────────────

    def _layout_trio(self, ratios: List[float], style: LayoutStyle,
                      rng: random.Random) -> List[GeneratedCell]:
        """Three images — hero + supporting or triptych."""
        variant = rng.choice(["hero_left", "hero_right", "hero_top", "triptych"])

        if style == LayoutStyle.MINIMAL:
            variant = "triptych"
        elif style in (LayoutStyle.MAGAZINE, LayoutStyle.DYNAMIC):
            variant = rng.choice(["hero_left", "hero_right", "hero_top"])

        # Golden ratio for hero size
        hero = 1.0 / PHI + 0.1  # ≈ 0.72
        support = 1.0 - hero

        if variant == "hero_left":
            return [
                GeneratedCell(0.0, 0.0, hero, 1.0, importance=1.5),
                GeneratedCell(hero, 0.0, support, 0.5, importance=0.8),
                GeneratedCell(hero, 0.5, support, 0.5, importance=0.8),
            ]
        elif variant == "hero_right":
            return [
                GeneratedCell(0.0, 0.0, support, 0.5, importance=0.8),
                GeneratedCell(0.0, 0.5, support, 0.5, importance=0.8),
                GeneratedCell(support, 0.0, hero, 1.0, importance=1.5),
            ]
        elif variant == "hero_top":
            return [
                GeneratedCell(0.0, 0.0, 1.0, hero, importance=1.5),
                GeneratedCell(0.0, hero, 0.5, support, importance=0.8),
                GeneratedCell(0.5, hero, 0.5, support, importance=0.8),
            ]
        else:  # triptych
            third = 1.0 / 3
            return [
                GeneratedCell(0.0, 0.0, third, 1.0, importance=1.0),
                GeneratedCell(third, 0.0, third, 1.0, importance=1.0),
                GeneratedCell(2 * third, 0.0, third, 1.0, importance=1.0),
            ]

    # ── Quad Layouts ───────────────────────────────────────────

    def _layout_quad(self, ratios: List[float], style: LayoutStyle,
                      rng: random.Random) -> List[GeneratedCell]:
        """Four images — grid, hero + 3, or L-shape."""
        variant = rng.choice(["grid", "hero_top_3", "hero_left_3", "l_shape"])

        if style == LayoutStyle.MINIMAL:
            variant = "grid"
        elif style == LayoutStyle.MAGAZINE:
            variant = rng.choice(["hero_top_3", "hero_left_3"])

        if variant == "grid":
            # 2×2 with optional golden ratio sizing
            if style == LayoutStyle.DYNAMIC:
                split_x = 1.0 / PHI
                split_y = 1.0 / PHI
            else:
                split_x = 0.5
                split_y = 0.5
            return [
                GeneratedCell(0.0, 0.0, split_x, split_y, importance=1.1),
                GeneratedCell(split_x, 0.0, 1.0 - split_x, split_y, importance=0.9),
                GeneratedCell(0.0, split_y, split_x, 1.0 - split_y, importance=0.9),
                GeneratedCell(split_x, split_y, 1.0 - split_x, 1.0 - split_y, importance=1.1),
            ]

        elif variant == "hero_top_3":
            hero_h = 0.62  # Golden ratio
            support_h = 1.0 - hero_h
            third = 1.0 / 3
            return [
                GeneratedCell(0.0, 0.0, 1.0, hero_h, importance=1.5),
                GeneratedCell(0.0, hero_h, third, support_h, importance=0.8),
                GeneratedCell(third, hero_h, third, support_h, importance=0.8),
                GeneratedCell(2 * third, hero_h, third, support_h, importance=0.8),
            ]

        elif variant == "hero_left_3":
            hero_w = 0.62
            support_w = 1.0 - hero_w
            third = 1.0 / 3
            return [
                GeneratedCell(0.0, 0.0, hero_w, 1.0, importance=1.5),
                GeneratedCell(hero_w, 0.0, support_w, third, importance=0.8),
                GeneratedCell(hero_w, third, support_w, third, importance=0.8),
                GeneratedCell(hero_w, 2 * third, support_w, third, importance=0.8),
            ]

        else:  # l_shape
            return [
                GeneratedCell(0.0, 0.0, 0.65, 0.65, importance=1.5),
                GeneratedCell(0.65, 0.0, 0.35, 0.45, importance=0.9),
                GeneratedCell(0.65, 0.45, 0.35, 0.55, importance=0.9),
                GeneratedCell(0.0, 0.65, 0.65, 0.35, importance=0.9),
            ]

    # ── Quint Layouts ──────────────────────────────────────────

    def _layout_quint(self, ratios: List[float], style: LayoutStyle,
                       rng: random.Random) -> List[GeneratedCell]:
        """Five images — mosaic, cross, or filmstrip."""
        variant = rng.choice(["mosaic_2_3", "hero_mosaic", "cross"])

        if style == LayoutStyle.MINIMAL:
            variant = "mosaic_2_3"
        elif style == LayoutStyle.MAGAZINE:
            variant = "hero_mosaic"

        if variant == "mosaic_2_3":
            # 2 on top, 3 on bottom
            return [
                GeneratedCell(0.0, 0.0, 0.50, 0.50, importance=1.1),
                GeneratedCell(0.50, 0.0, 0.50, 0.50, importance=1.1),
                GeneratedCell(0.0, 0.50, 0.333, 0.50, importance=0.9),
                GeneratedCell(0.333, 0.50, 0.334, 0.50, importance=0.9),
                GeneratedCell(0.667, 0.50, 0.333, 0.50, importance=0.9),
            ]

        elif variant == "hero_mosaic":
            # Big hero + 4 small tiles
            return [
                GeneratedCell(0.0, 0.0, 0.62, 1.0, importance=1.8),
                GeneratedCell(0.62, 0.0, 0.38, 0.25, importance=0.7),
                GeneratedCell(0.62, 0.25, 0.38, 0.25, importance=0.7),
                GeneratedCell(0.62, 0.50, 0.38, 0.25, importance=0.7),
                GeneratedCell(0.62, 0.75, 0.38, 0.25, importance=0.7),
            ]

        else:  # cross
            # Center hero + 4 corners
            cx, cy = 0.25, 0.20
            cw, ch = 0.50, 0.60
            return [
                GeneratedCell(cx, cy, cw, ch, importance=1.8),
                GeneratedCell(0.0, 0.0, cx, cy + ch * 0.5, importance=0.7),
                GeneratedCell(cx + cw, 0.0, 1.0 - cx - cw, cy + ch * 0.5, importance=0.7),
                GeneratedCell(0.0, cy + ch * 0.5, cx, 1.0 - cy - ch * 0.5, importance=0.7),
                GeneratedCell(cx + cw, cy + ch * 0.5, 1.0 - cx - cw, 1.0 - cy - ch * 0.5, importance=0.7),
            ]

    # ── Grid Layout (6+) ──────────────────────────────────────

    def _layout_grid(self, count: int, ratios: List[float],
                      style: LayoutStyle, rng: random.Random) -> List[GeneratedCell]:
        """6+ images — intelligent grid with optional hero."""
        if style == LayoutStyle.MAGAZINE and count >= 6:
            # Hero + grid for the rest
            hero_w = 0.50
            hero_h = 0.50
            cells = [GeneratedCell(0.0, 0.0, hero_w, hero_h, importance=1.8)]
            remaining = count - 1
            # Fill remaining space with grid
            grid_cells = self._fill_grid_region(
                remaining, hero_w, 0.0, 1.0 - hero_w, hero_h
            )
            grid_cells += self._fill_grid_region(
                max(0, remaining - len(grid_cells)),
                0.0, hero_h, 1.0, 1.0 - hero_h
            )
            cells.extend(grid_cells[:remaining])
            return cells

        # Standard grid
        cols = math.ceil(math.sqrt(count * 1.5))  # Landscape-biased
        rows = math.ceil(count / cols)

        cells = []
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows
        idx = 0

        for r in range(rows):
            for c in range(cols):
                if idx >= count:
                    break
                cells.append(GeneratedCell(
                    x=c * cell_w,
                    y=r * cell_h,
                    w=cell_w,
                    h=cell_h,
                    importance=1.0,
                ))
                idx += 1

        return cells

    def _fill_grid_region(self, count: int, x: float, y: float,
                           w: float, h: float) -> List[GeneratedCell]:
        """Fill a rectangular region with a grid of cells."""
        if count <= 0 or w <= 0 or h <= 0:
            return []

        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        cell_w = w / cols
        cell_h = h / rows

        cells = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= count:
                    break
                cells.append(GeneratedCell(
                    x=x + c * cell_w,
                    y=y + r * cell_h,
                    w=cell_w,
                    h=cell_h,
                    importance=0.8,
                ))
                idx += 1
        return cells

    # ── Style Modifiers ────────────────────────────────────────

    def _apply_style_modifiers(self, cells: List[GeneratedCell],
                                style: LayoutStyle,
                                rng: random.Random) -> List[GeneratedCell]:
        """Apply style-specific adjustments to generated cells."""
        if style == LayoutStyle.ELEGANT:
            # Add inner padding to each cell (5% inset)
            inset = 0.02
            cells = [
                GeneratedCell(
                    c.x + inset, c.y + inset,
                    max(c.w - 2 * inset, 0.05),
                    max(c.h - 2 * inset, 0.05),
                    c.importance
                ) for c in cells
            ]

        elif style == LayoutStyle.DYNAMIC:
            # Slightly randomize positions for organic feel
            # Scale jitter inversely with count to prevent overlap in dense grids
            base_jitter = 0.01
            jitter = base_jitter / max(len(cells), 1)
            for c in cells:
                c.x = max(0, min(c.x + rng.uniform(-jitter, jitter), 1.0 - c.w))
                c.y = max(0, min(c.y + rng.uniform(-jitter, jitter), 1.0 - c.h))

        elif style == LayoutStyle.MINIMAL:
            # Generous uniform padding
            inset = 0.03
            cells = [
                GeneratedCell(
                    c.x + inset, c.y + inset,
                    max(c.w - 2 * inset, 0.05),
                    max(c.h - 2 * inset, 0.05),
                    c.importance
                ) for c in cells
            ]

        return cells

    # ── Layout Scoring ─────────────────────────────────────────

    def _score_layout(self, cells: List[GeneratedCell],
                       ratios: List[float],
                       page_aspect: float) -> float:
        """
        Score a layout based on compositional quality.
        Higher score = better layout.

        Criteria:
        1. Coverage — how much of the page is used (60–90% is ideal)
        2. Aspect match — cells match image aspect ratios
        3. Balance — visual center of mass near page center
        4. Overlap penalty — overlapping cells are bad
        """
        if not cells:
            return 0.0

        score = 100.0

        # 1. Coverage score (0–25 points)
        total_area = sum(c.w * c.h for c in cells)
        if 0.6 <= total_area <= 0.95:
            coverage_score = 25.0
        elif total_area > 0.95:
            coverage_score = 25.0 - (total_area - 0.95) * 100
        else:
            coverage_score = total_area / 0.6 * 25.0
        score += max(0, coverage_score)

        # 2. Aspect ratio match (0–25 points)
        aspect_score = 0
        for i, cell in enumerate(cells):
            if i < len(ratios):
                cell_ratio = (cell.w * page_aspect) / max(cell.h, 0.01)
                image_ratio = ratios[i]
                match = 1.0 - min(abs(cell_ratio - image_ratio) / max(image_ratio, 0.01), 1.0)
                aspect_score += match
        if cells:
            aspect_score = (aspect_score / len(cells)) * 25.0
        score += aspect_score

        # 3. Balance score (0–25 points)
        if cells:
            weights = [c.w * c.h * c.importance for c in cells]
            total_weight = sum(weights)
            if total_weight > 0:
                cx = sum((c.x + c.w / 2) * w for c, w in zip(cells, weights)) / total_weight
                cy = sum((c.y + c.h / 2) * w for c, w in zip(cells, weights)) / total_weight
                dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
                balance_score = 25.0 * (1.0 - min(dist * 4, 1.0))
                score += balance_score

        # 4. Overlap penalty (-50 per overlap)
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                if self._cells_overlap(cells[i], cells[j]):
                    score -= 50.0

        return max(0, score)

    def _cells_overlap(self, a: GeneratedCell, b: GeneratedCell) -> bool:
        """Check if two cells overlap (with small tolerance)."""
        tol = 0.005
        return not (
            a.x + a.w <= b.x + tol or
            b.x + b.w <= a.x + tol or
            a.y + a.h <= b.y + tol or
            b.y + b.h <= a.y + tol
        )
