"""
AI Layout Generator (Phase 3 — Rule-Based)
============================================
Generates visually balanced album page layouts using mathematical
optimization. No ML model required — uses compositional principles:

  1. Golden Ratio Partitioning (φ = 1.618)
  2. Visual Weight Balancing (hero images get more area)
  3. Aspect Ratio Matching (tall images → tall cells)
  4. Style-aware modifiers (elegant, dynamic, minimal, classic, magazine)
  5. Seed-based reproducibility (per-page variation)

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
    importance: float = 1.0
    prefers_portrait: bool = False   # Hint: assign portrait images here


@dataclass
class GeneratedLayout:
    """A complete generated layout with metadata."""
    name: str
    cells: List[GeneratedCell]
    style: str
    score: float
    seed: int
    image_indices: Optional[List[int]] = None  # Maps cell[i] -> image[i]


# ────────────────────────────────────────────────────────────────
# Structural Templates — Pool of layout structures to cycle through
# Each returns normalized cells. The AI picks the best variant per page.
# ────────────────────────────────────────────────────────────────

def _make_structures_for_count(count: int, page_aspect: float) -> List[str]:
    """Return a list of structure-generator names for a given count."""
    structures = {
        1: ["full_bleed", "centered"],
        2: ["split_h", "split_v", "golden_h", "golden_v"],
        3: ["hero_left", "hero_right", "hero_top", "triptych", "hero_bottom"],
        4: ["grid_2x2", "hero_top_3", "hero_left_3", "l_shape", "golden_grid"],
        5: ["mosaic_2_3", "hero_mosaic", "mosaic_3_2", "filmstrip_top", "asymmetric_5"],
        6: ["grid_2x3", "grid_3x2", "hero_grid_5", "big2_small4", "mosaic_6"],
    }
    if count <= 6:
        return structures.get(count, ["full_bleed"])
    else:
        return ["smart_grid"]


class AILayoutGenerator:
    """
    Rule-based layout generator using compositional mathematics.
    Generates multiple layout variants and scores them.
    Uses page_number to vary layouts across pages.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, image_count: int,
                 aspect_ratios: Optional[List[float]] = None,
                 page_aspect: float = 1.5,
                 style: str = "classic",
                 num_variants: int = 3,
                 page_number: int = 0) -> GeneratedLayout:
        """
        Generate the best layout for a set of images.
        Uses deterministic round-robin cycling through structures
        to ensure every page looks different.
        """
        if aspect_ratios is None:
            aspect_ratios = [1.5] * image_count
        while len(aspect_ratios) < image_count:
            aspect_ratios.append(1.5)
        aspect_ratios = aspect_ratios[:image_count]

        style_enum = LayoutStyle(style) if style in [s.value for s in LayoutStyle] else LayoutStyle.CLASSIC

        # Get ALL available structures for this image count
        structures = _make_structures_for_count(image_count, page_aspect)

        # Deterministic round-robin: page_number selects the structure
        structure_idx = page_number % len(structures)
        structure_name = structures[structure_idx]

        # Build the layout with this structure
        page_seed = self.seed + page_number * 31
        rng = random.Random(page_seed)
        cells = self._build_structure(structure_name, image_count, aspect_ratios, style_enum, rng)
        cells = self._apply_style_modifiers(cells, style_enum, rng)
        score, indices = self._score_layout_and_assign(cells, aspect_ratios, page_aspect)

        name = f"ai_{style}_{image_count}img_{structure_name}_s{page_seed}"
        return GeneratedLayout(
            name=name, cells=cells, style=style,
            score=score, seed=page_seed,
            image_indices=indices,
        )

    def generate_all_variants(self, image_count: int,
                               aspect_ratios: Optional[List[float]] = None,
                               page_aspect: float = 1.5,
                               style: str = "classic",
                               num_variants: int = 5,
                               page_number: int = 0) -> List[GeneratedLayout]:
        """Generate ALL structure variants for a given count (sorted by score)."""
        if aspect_ratios is None:
            aspect_ratios = [1.5] * image_count
        while len(aspect_ratios) < image_count:
            aspect_ratios.append(1.5)
        aspect_ratios = aspect_ratios[:image_count]

        style_enum = LayoutStyle(style) if style in [s.value for s in LayoutStyle] else LayoutStyle.CLASSIC
        structures = _make_structures_for_count(image_count, page_aspect)

        variants = []
        for i, sname in enumerate(structures):
            variant_seed = self.seed + page_number * 31 + i * 97
            rng = random.Random(variant_seed)
            cells = self._build_structure(sname, image_count, aspect_ratios, style_enum, rng)
            cells = self._apply_style_modifiers(cells, style_enum, rng)
            score = self._score_layout(cells, aspect_ratios, page_aspect)

            name = f"ai_{style}_{image_count}img_{sname}_s{variant_seed}"
            variants.append(GeneratedLayout(
                name=name, cells=cells, style=style,
                score=score, seed=variant_seed,
            ))

        variants.sort(key=lambda v: v.score, reverse=True)
        return variants

    def _generate_variant(self, count: int, ratios: List[float],
                           page_aspect: float, style: str,
                           rng: random.Random, seed: int) -> GeneratedLayout:
        """Generate a single layout variant (used internally)."""
        style_enum = LayoutStyle(style) if style in [s.value for s in LayoutStyle] else LayoutStyle.CLASSIC
        structures = _make_structures_for_count(count, page_aspect)
        structure_name = rng.choice(structures)
        cells = self._build_structure(structure_name, count, ratios, style_enum, rng)
        cells = self._apply_style_modifiers(cells, style_enum, rng)

        # Find best assignment of images to cells
        score, indices = self._score_layout_and_assign(cells, ratios, page_aspect)

        name = f"ai_{style}_{count}img_{structure_name}_s{seed}"
        return GeneratedLayout(
            name=name, cells=cells, style=style,
            score=score, seed=seed,
            image_indices=indices,
        )

    def _build_structure(self, name: str, count: int, ratios: List[float],
                          style: LayoutStyle, rng: random.Random) -> List[GeneratedCell]:
        """Build a specific layout structure."""

        # ── Single image ──
        if name == "full_bleed":
            return [GeneratedCell(0.0, 0.0, 1.0, 1.0)]
        if name == "centered":
            pad = 0.08
            return [GeneratedCell(pad, pad, 1.0 - 2 * pad, 1.0 - 2 * pad)]

        # ── Two images ──
        if name == "split_h":
            return [
                GeneratedCell(0.0, 0.0, 0.48, 1.0),
                GeneratedCell(0.52, 0.0, 0.48, 1.0),
            ]
        if name == "split_v":
            return [
                GeneratedCell(0.0, 0.0, 1.0, 0.48),
                GeneratedCell(0.0, 0.52, 1.0, 0.48),
            ]
        if name == "golden_h":
            s = 0.60
            return [
                GeneratedCell(0.0, 0.0, s - 0.02, 1.0, importance=1.3),
                GeneratedCell(s + 0.02, 0.0, 1.0 - s - 0.02, 1.0, importance=0.9),
            ]
        if name == "golden_v":
            s = 0.60
            return [
                GeneratedCell(0.0, 0.0, 1.0, s - 0.02, importance=1.3),
                GeneratedCell(0.0, s + 0.02, 1.0, 1.0 - s - 0.02, importance=0.9),
            ]

        # ── Three images ──
        if name == "hero_left":
            g = 0.03
            hw = 0.62
            return [
                GeneratedCell(0.0, 0.0, hw - g, 1.0, importance=1.5),
                GeneratedCell(hw + g, 0.0, 1.0 - hw - g, 0.48, importance=0.8, prefers_portrait=True),
                GeneratedCell(hw + g, 0.52, 1.0 - hw - g, 0.48, importance=0.8, prefers_portrait=True),
            ]
        if name == "hero_right":
            g = 0.03
            sw = 0.35
            return [
                GeneratedCell(0.0, 0.0, sw - g, 0.48, importance=0.8, prefers_portrait=True),
                GeneratedCell(0.0, 0.52, sw - g, 0.48, importance=0.8, prefers_portrait=True),
                GeneratedCell(sw + g, 0.0, 1.0 - sw - g, 1.0, importance=1.5),
            ]
        if name == "hero_top":
            g = 0.03
            hh = 0.60
            return [
                GeneratedCell(0.0, 0.0, 1.0, hh - g, importance=1.5),
                GeneratedCell(0.0, hh + g, 0.48, 1.0 - hh - g, importance=0.8),
                GeneratedCell(0.52, hh + g, 0.48, 1.0 - hh - g, importance=0.8),
            ]
        if name == "hero_bottom":
            g = 0.03
            sh = 0.37
            return [
                GeneratedCell(0.0, 0.0, 0.48, sh - g, importance=0.8),
                GeneratedCell(0.52, 0.0, 0.48, sh - g, importance=0.8),
                GeneratedCell(0.0, sh + g, 1.0, 1.0 - sh - g, importance=1.5),
            ]
        if name == "triptych":
            g = 0.02
            w = (1.0 - 2 * g) / 3
            return [
                GeneratedCell(0.0, 0.0, w, 1.0),
                GeneratedCell(w + g, 0.0, w, 1.0),
                GeneratedCell(2 * (w + g), 0.0, w, 1.0),
            ]

        # ── Four images ──
        if name == "grid_2x2":
            g = 0.03
            w = (1.0 - g) / 2
            h = (1.0 - g) / 2
            return [
                GeneratedCell(0.0, 0.0, w, h),
                GeneratedCell(w + g, 0.0, w, h),
                GeneratedCell(0.0, h + g, w, h),
                GeneratedCell(w + g, h + g, w, h),
            ]
        if name == "hero_top_3":
            g = 0.03
            hh = 0.60
            tw = (1.0 - 2 * g) / 3
            return [
                GeneratedCell(0.0, 0.0, 1.0, hh - g, importance=1.5),
                GeneratedCell(0.0, hh + g, tw, 1.0 - hh - g, importance=0.8),
                GeneratedCell(tw + g, hh + g, tw, 1.0 - hh - g, importance=0.8),
                GeneratedCell(2 * (tw + g), hh + g, tw, 1.0 - hh - g, importance=0.8),
            ]
        if name == "hero_left_3":
            g = 0.03
            hw = 0.60
            th = (1.0 - 2 * g) / 3
            return [
                GeneratedCell(0.0, 0.0, hw - g, 1.0, importance=1.5),
                GeneratedCell(hw + g, 0.0, 1.0 - hw - g, th, importance=0.8, prefers_portrait=True),
                GeneratedCell(hw + g, th + g, 1.0 - hw - g, th, importance=0.8, prefers_portrait=True),
                GeneratedCell(hw + g, 2 * (th + g), 1.0 - hw - g, th, importance=0.8, prefers_portrait=True),
            ]
        if name == "l_shape":
            g = 0.03
            return [
                GeneratedCell(0.0, 0.0, 0.62, 0.62, importance=1.5),
                GeneratedCell(0.62 + g, 0.0, 1.0 - 0.62 - g, 0.48, importance=0.9),
                GeneratedCell(0.62 + g, 0.48 + g, 1.0 - 0.62 - g, 0.52 - g, importance=0.9),
                GeneratedCell(0.0, 0.62 + g, 0.62, 1.0 - 0.62 - g, importance=0.9),
            ]
        if name == "golden_grid":
            g = 0.03
            sw = 1.0 / PHI
            return [
                GeneratedCell(0.0, 0.0, sw - g, 0.48, importance=1.1),
                GeneratedCell(sw + g, 0.0, 1.0 - sw - g, 0.48, importance=0.9),
                GeneratedCell(0.0, 0.48 + g, sw - g, 0.52 - g, importance=0.9),
                GeneratedCell(sw + g, 0.48 + g, 1.0 - sw - g, 0.52 - g, importance=1.1),
            ]

        # ── Five images ──
        if name == "mosaic_2_3":
            g = 0.03
            tw = (1.0 - 2 * g) / 3
            hw = (1.0 - g) / 2
            return [
                GeneratedCell(0.0, 0.0, hw, 0.48, importance=1.1),
                GeneratedCell(hw + g, 0.0, hw, 0.48, importance=1.1),
                GeneratedCell(0.0, 0.48 + g, tw, 0.52 - g, importance=0.9),
                GeneratedCell(tw + g, 0.48 + g, tw, 0.52 - g, importance=0.9),
                GeneratedCell(2 * (tw + g), 0.48 + g, tw, 0.52 - g, importance=0.9),
            ]
        if name == "mosaic_3_2":
            g = 0.03
            tw = (1.0 - 2 * g) / 3
            hw = (1.0 - g) / 2
            return [
                GeneratedCell(0.0, 0.0, tw, 0.48, importance=0.9),
                GeneratedCell(tw + g, 0.0, tw, 0.48, importance=0.9),
                GeneratedCell(2 * (tw + g), 0.0, tw, 0.48, importance=0.9),
                GeneratedCell(0.0, 0.48 + g, hw, 0.52 - g, importance=1.1),
                GeneratedCell(hw + g, 0.48 + g, hw, 0.52 - g, importance=1.1),
            ]
        if name == "hero_mosaic":
            g = 0.03
            hw = 0.58
            qh = (1.0 - 3 * g) / 4
            return [
                GeneratedCell(0.0, 0.0, hw - g, 1.0, importance=1.8),
                GeneratedCell(hw + g, 0.0, 1.0 - hw - g, qh, importance=0.7),
                GeneratedCell(hw + g, qh + g, 1.0 - hw - g, qh, importance=0.7),
                GeneratedCell(hw + g, 2 * (qh + g), 1.0 - hw - g, qh, importance=0.7),
                GeneratedCell(hw + g, 3 * (qh + g), 1.0 - hw - g, qh, importance=0.7),
            ]
        if name == "filmstrip_top":
            g = 0.03
            tw = (1.0 - 3 * g) / 4
            return [
                GeneratedCell(0.0, 0.0, tw, 0.35, importance=0.8),
                GeneratedCell(tw + g, 0.0, tw, 0.35, importance=0.8),
                GeneratedCell(2 * (tw + g), 0.0, tw, 0.35, importance=0.8),
                GeneratedCell(3 * (tw + g), 0.0, tw, 0.35, importance=0.8),
                GeneratedCell(0.0, 0.35 + g, 1.0, 0.65 - g, importance=1.5),
            ]
        if name == "asymmetric_5":
            g = 0.03
            return [
                GeneratedCell(0.0, 0.0, 0.48, 0.58, importance=1.3),
                GeneratedCell(0.48 + g, 0.0, 0.52 - g, 0.38, importance=1.0),
                GeneratedCell(0.48 + g, 0.38 + g, 0.52 - g, 0.62 - g, importance=1.0),
                GeneratedCell(0.0, 0.58 + g, 0.30, 0.42 - g, importance=0.8),
                GeneratedCell(0.30 + g, 0.58 + g, 0.18 - g, 0.42 - g, importance=0.8),
            ]

        # ── Six images ──
        if name == "grid_2x3":
            g = 0.03
            tw = (1.0 - 2 * g) / 3
            th = (1.0 - g) / 2
            return [
                GeneratedCell(0.0, 0.0, tw, th, importance=1.0),
                GeneratedCell(tw + g, 0.0, tw, th, importance=1.0),
                GeneratedCell(2 * (tw + g), 0.0, tw, th, importance=1.0),
                GeneratedCell(0.0, th + g, tw, th, importance=1.0),
                GeneratedCell(tw + g, th + g, tw, th, importance=1.0),
                GeneratedCell(2 * (tw + g), th + g, tw, th, importance=1.0),
            ]
        if name == "grid_3x2":
            g = 0.03
            tw = (1.0 - g) / 2
            th = (1.0 - 2 * g) / 3
            return [
                GeneratedCell(0.0, 0.0, tw, th, importance=1.0),
                GeneratedCell(tw + g, 0.0, tw, th, importance=1.0),
                GeneratedCell(0.0, th + g, tw, th, importance=1.0),
                GeneratedCell(tw + g, th + g, tw, th, importance=1.0),
                GeneratedCell(0.0, 2 * (th + g), tw, th, importance=1.0),
                GeneratedCell(tw + g, 2 * (th + g), tw, th, importance=1.0),
            ]
        if name == "hero_grid_5":
            g = 0.03
            hw = 0.55
            sw = 1.0 - hw - g
            sh = (1.0 - g) / 2
            qw = (sw - g) / 2
            return [
                GeneratedCell(0.0, 0.0, hw - g, 1.0, importance=1.5),
                GeneratedCell(hw + g, 0.0, qw, sh, importance=0.8),
                GeneratedCell(hw + g + qw + g, 0.0, qw, sh, importance=0.8),
                GeneratedCell(hw + g, sh + g, sw, sh, importance=1.0),
                # Split bottom-right into 2
                GeneratedCell(hw + g, sh + g, qw, sh, importance=0.8),
                GeneratedCell(hw + g + qw + g, sh + g, qw, sh, importance=0.8),
            ]
        if name == "big2_small4":
            g = 0.03
            bw = (1.0 - g) / 2
            bh = 0.58
            sw = (1.0 - g) / 2
            sh = (1.0 - bh - 2 * g) / 1
            qw = (1.0 - 3 * g) / 4
            return [
                GeneratedCell(0.0, 0.0, bw, bh, importance=1.3),
                GeneratedCell(bw + g, 0.0, bw, bh, importance=1.3),
                GeneratedCell(0.0, bh + g, qw, 1.0 - bh - g, importance=0.8),
                GeneratedCell(qw + g, bh + g, qw, 1.0 - bh - g, importance=0.8),
                GeneratedCell(2 * (qw + g), bh + g, qw, 1.0 - bh - g, importance=0.8),
                GeneratedCell(3 * (qw + g), bh + g, qw, 1.0 - bh - g, importance=0.8),
            ]
        if name == "mosaic_6":
            g = 0.03
            return [
                GeneratedCell(0.0, 0.0, 0.38, 0.55, importance=1.2),
                GeneratedCell(0.38 + g, 0.0, 0.30, 0.35, importance=0.9),
                GeneratedCell(0.68 + g, 0.0, 0.32 - g, 0.55, importance=1.0, prefers_portrait=True),
                GeneratedCell(0.38 + g, 0.35 + g, 0.30, 0.65 - g, importance=1.0, prefers_portrait=True),
                GeneratedCell(0.0, 0.55 + g, 0.38, 0.45 - g, importance=0.9),
                GeneratedCell(0.68 + g, 0.55 + g, 0.32 - g, 0.45 - g, importance=0.9),
            ]

        # ── Fallback for 7+ images: smart grid ──
        return self._smart_grid(count, ratios, rng)

    def _smart_grid(self, count: int, ratios: List[float],
                     rng: random.Random) -> List[GeneratedCell]:
        """Intelligent grid for 7+ images with built-in gaps."""
        g = 0.03
        cols = math.ceil(math.sqrt(count * 1.5))
        rows = math.ceil(count / cols)

        cell_w = (1.0 - (cols - 1) * g) / cols
        cell_h = (1.0 - (rows - 1) * g) / rows

        cells = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= count:
                    break
                cells.append(GeneratedCell(
                    x=c * (cell_w + g),
                    y=r * (cell_h + g),
                    w=cell_w,
                    h=cell_h,
                ))
                idx += 1
        return cells

    # ── Style Modifiers ────────────────────────────────────────

    def _apply_style_modifiers(self, cells: List[GeneratedCell],
                                style: LayoutStyle,
                                rng: random.Random) -> List[GeneratedCell]:
        """Apply style-specific adjustments."""
        if style == LayoutStyle.ELEGANT:
            inset = 0.02
            cells = [
                GeneratedCell(
                    c.x + inset, c.y + inset,
                    max(c.w - 2 * inset, 0.05),
                    max(c.h - 2 * inset, 0.05),
                    c.importance, c.prefers_portrait
                ) for c in cells
            ]

        elif style == LayoutStyle.DYNAMIC:
            base_jitter = 0.008
            jitter = base_jitter / max(len(cells), 1)
            for c in cells:
                c.x = max(0, min(c.x + rng.uniform(-jitter, jitter), 1.0 - c.w))
                c.y = max(0, min(c.y + rng.uniform(-jitter, jitter), 1.0 - c.h))

        elif style == LayoutStyle.MINIMAL:
            inset = 0.02
            cells = [
                GeneratedCell(
                    c.x + inset, c.y + inset,
                    max(c.w - 2 * inset, 0.05),
                    max(c.h - 2 * inset, 0.05),
                    c.importance, c.prefers_portrait
                ) for c in cells
            ]

        return cells

    # ── Layout Scoring ─────────────────────────────────────────

    # ── Layout Scoring & Assignment ────────────────────────────

    def _score_layout_and_assign(self, cells: List[GeneratedCell],
                                  ratios: List[float],
                                  page_aspect: float) -> Tuple[float, List[int]]:
        """
        Score a layout based on compositional quality, BUT find the optimal
        assignment of images to cells first (based on aspect ratio rank).

        Returns:
            (best_score, best_image_indices)
        """
        if not cells:
            return 0.0, []

        # 1. Rank-order matching for aspect ratios
        # Sort cells by aspect ratio (wide -> tall)
        cell_ratios = []
        for i, c in enumerate(cells):
            ar = (c.w * page_aspect) / max(c.h, 0.01)
            cell_ratios.append((ar, i))
        
        # Sort images by aspect ratio (wide -> tall)
        img_ratios = []
        for i, r in enumerate(ratios):
            img_ratios.append((r, i))
        
        # Sort both descending (widest first)
        cell_ratios.sort(key=lambda x: x[0], reverse=True)
        img_ratios.sort(key=lambda x: x[0], reverse=True)
        
        # Create assignment map: cell_index -> image_index
        # We need to map: layout.cells[k] should take image ratios[assigned_idx]
        # But the caller needs: image_indices such that cells[k] gets image[image_indices[k]]
        
        # Mapping: valid only if counts match. If diff, just truncate/pad.
        n = min(len(cells), len(ratios))
        
        # assigned_images[cell_idx] = image_idx
        assigned_images = list(range(len(cells))) # Default identity
        
        for k in range(n):
            c_idx = cell_ratios[k][1]
            i_idx = img_ratios[k][1]
            assigned_images[c_idx] = i_idx
            
        # 2. Compute score with this optimal assignment
        score = 100.0

        # Coverage score (0–25)
        total_area = sum(c.w * c.h for c in cells)
        if 0.55 <= total_area <= 0.92:
            coverage_score = 25.0
        elif total_area > 0.92:
            coverage_score = 25.0 - (total_area - 0.92) * 80
        else:
            coverage_score = total_area / 0.55 * 25.0
        score += max(0, coverage_score)

        # Aspect ratio match score (0–50) with optimized assignment
        aspect_score = 0
        for c_idx, i_idx in enumerate(assigned_images):
            if i_idx < len(ratios):
                cell = cells[c_idx]
                cell_ratio = (cell.w * page_aspect) / max(cell.h, 0.01)
                image_ratio = ratios[i_idx]
                match = 1.0 - min(abs(cell_ratio - image_ratio) / max(image_ratio, 0.01), 1.0)
                aspect_score += match
        
        if cells:
            # Boost aspect score weight since we are optimizing for it
            aspect_score = (aspect_score / len(cells)) * 50.0
        score += aspect_score

        # Balance score (0–25)
        if cells:
            weights = [c.w * c.h * c.importance for c in cells]
            total_weight = sum(weights)
            if total_weight > 0:
                cx = sum((c.x + c.w / 2) * w for c, w in zip(cells, weights)) / total_weight
                cy = sum((c.y + c.h / 2) * w for c, w in zip(cells, weights)) / total_weight
                dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
                balance_score = 25.0 * (1.0 - min(dist * 4, 1.0))
                score += balance_score

        # Overlap penalty
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                if self._cells_overlap(cells[i], cells[j]):
                    score -= 50.0

        # Full-page usage bonus
        max_x = max(c.x + c.w for c in cells)
        max_y = max(c.y + c.h for c in cells)
        if max_x > 0.90 and max_y > 0.90:
            score += 10.0

        # Fold-crossing penalty: penalise cells that straddle x=0.5
        # (the center fold of the album spread).  A normalized fold
        # exclusion zone of ~3% on each side of center (0.47–0.53).
        fold_left = 0.47
        fold_right = 0.53
        for c in cells:
            c_left = c.x
            c_right = c.x + c.w
            if c_left < fold_left and c_right > fold_right:
                # Cell spans fully across the fold — heavy penalty
                score -= 30.0
            elif c_left < fold_right and c_right > fold_left:
                # Cell partially overlaps the fold margin
                score -= 15.0

        # print(f"DEBUG: AI Assign: {assigned_images} score={score}")
        return max(0, score), assigned_images

    # Retrofit old _score_layout to use new logic (it's called by generate_all_variants too)
    def _score_layout(self, cells, ratios, page_aspect):
        s, _ = self._score_layout_and_assign(cells, ratios, page_aspect)
        return s

    def _cells_overlap(self, a: GeneratedCell, b: GeneratedCell) -> bool:
        """Check if two cells overlap."""
        tol = 0.005
        return not (
            a.x + a.w <= b.x + tol or
            b.x + b.w <= a.x + tol or
            a.y + a.h <= b.y + tol or
            b.y + b.h <= a.y + tol
        )
