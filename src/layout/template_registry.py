"""
Template Registry
==================
Loads, validates, and serves JSON layout templates.
Ships with bundled defaults for common wedding album layouts.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TemplateCell:
    """Normalized cell placement (0.0–1.0 coordinate space)."""
    x: float
    y: float
    w: float
    h: float


@dataclass
class LayoutTemplate:
    """A single page layout template."""
    name: str
    description: str
    image_count: int
    cells: List[TemplateCell]


# ────────────────────────────────────────────────────────────────
# Bundled Default Templates
# ────────────────────────────────────────────────────────────────

BUNDLED_TEMPLATES: List[Dict] = [
    {
        "name": "single_full",
        "description": "Single image, full bleed",
        "image_count": 1,
        "cells": [
            {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
        ]
    },
    {
        "name": "split_horizontal",
        "description": "Two images side by side with fold gap",
        "image_count": 2,
        "cells": [
            {"x": 0.0, "y": 0.0, "w": 0.47, "h": 1.0},
            {"x": 0.53, "y": 0.0, "w": 0.47, "h": 1.0},
        ]
    },
    {
        "name": "split_vertical",
        "description": "Two images stacked vertically",
        "image_count": 2,
        "cells": [
            {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.50},
            {"x": 0.0, "y": 0.50, "w": 1.0, "h": 0.50},
        ]
    },
    {
        "name": "triptych",
        "description": "Three equal vertical panels",
        "image_count": 3,
        "cells": [
            {"x": 0.0,   "y": 0.0, "w": 0.333, "h": 1.0},
            {"x": 0.333, "y": 0.0, "w": 0.334, "h": 1.0},
            {"x": 0.667, "y": 0.0, "w": 0.333, "h": 1.0},
        ]
    },
    {
        "name": "hero_right_2",
        "description": "Hero image on the right, two stacked on the left",
        "image_count": 3,
        "cells": [
            {"x": 0.0,  "y": 0.0,  "w": 0.40, "h": 0.50},
            {"x": 0.0,  "y": 0.50, "w": 0.40, "h": 0.50},
            {"x": 0.40, "y": 0.0,  "w": 0.60, "h": 1.0},
        ]
    },
    {
        "name": "hero_left_2",
        "description": "Hero image on the left, two stacked on the right",
        "image_count": 3,
        "cells": [
            {"x": 0.0,  "y": 0.0,  "w": 0.60, "h": 1.0},
            {"x": 0.60, "y": 0.0,  "w": 0.40, "h": 0.50},
            {"x": 0.60, "y": 0.50, "w": 0.40, "h": 0.50},
        ]
    },
    {
        "name": "grid_2x2",
        "description": "Four quadrants with fold gap",
        "image_count": 4,
        "cells": [
            {"x": 0.0,  "y": 0.0,  "w": 0.47, "h": 0.50},
            {"x": 0.53, "y": 0.0,  "w": 0.47, "h": 0.50},
            {"x": 0.0,  "y": 0.50, "w": 0.47, "h": 0.50},
            {"x": 0.53, "y": 0.50, "w": 0.47, "h": 0.50},
        ]
    },
    {
        "name": "hero_top_3",
        "description": "Hero image on top, three smaller images below",
        "image_count": 4,
        "cells": [
            {"x": 0.0,   "y": 0.0,   "w": 1.0,   "h": 0.60},
            {"x": 0.0,   "y": 0.60,  "w": 0.333, "h": 0.40},
            {"x": 0.333, "y": 0.60,  "w": 0.334, "h": 0.40},
            {"x": 0.667, "y": 0.60,  "w": 0.333, "h": 0.40},
        ]
    },
    {
        "name": "grid_5_mosaic",
        "description": "Five-image mosaic: 2 top, 3 bottom with fold gap",
        "image_count": 5,
        "cells": [
            {"x": 0.0,   "y": 0.0,  "w": 0.47,  "h": 0.50},
            {"x": 0.53,  "y": 0.0,  "w": 0.47,  "h": 0.50},
            {"x": 0.0,   "y": 0.50, "w": 0.333, "h": 0.50},
            {"x": 0.333, "y": 0.50, "w": 0.334, "h": 0.50},
            {"x": 0.667, "y": 0.50, "w": 0.333, "h": 0.50},
        ]
    },
    {
        "name": "grid_2x3",
        "description": "Six images in 2 rows of 3",
        "image_count": 6,
        "cells": [
            {"x": 0.0,   "y": 0.0,  "w": 0.333, "h": 0.50},
            {"x": 0.333, "y": 0.0,  "w": 0.334, "h": 0.50},
            {"x": 0.667, "y": 0.0,  "w": 0.333, "h": 0.50},
            {"x": 0.0,   "y": 0.50, "w": 0.333, "h": 0.50},
            {"x": 0.333, "y": 0.50, "w": 0.334, "h": 0.50},
            {"x": 0.667, "y": 0.50, "w": 0.333, "h": 0.50},
        ]
    },
]


class TemplateRegistry:
    """
    Manages layout templates: bundled defaults + user-provided JSON files.
    Templates are indexed by image_count for fast lookup.
    """

    def __init__(self, user_template_dir: Optional[Path] = None):
        self._templates: Dict[str, LayoutTemplate] = {}
        self._by_count: Dict[int, List[LayoutTemplate]] = {}

        # Load bundled defaults
        for t_data in BUNDLED_TEMPLATES:
            self._register(t_data)

        # Load user templates from directory
        if user_template_dir and user_template_dir.exists():
            for f in sorted(user_template_dir.rglob("*.json")):
                try:
                    with open(f) as fp:
                        t_data = json.load(fp)
                    self._register(t_data)
                except Exception as e:
                    print(f"Warning: Failed to load template {f}: {e}")

    def _register(self, data: dict):
        """Parse and register a template."""
        cells = [TemplateCell(**c) for c in data["cells"]]
        template = LayoutTemplate(
            name=data["name"],
            description=data.get("description", ""),
            image_count=data["image_count"],
            cells=cells,
        )
        self._templates[template.name] = template
        count = template.image_count
        if count not in self._by_count:
            self._by_count[count] = []
        self._by_count[count].append(template)

    def get_by_name(self, name: str) -> Optional[LayoutTemplate]:
        """Get a template by exact name."""
        return self._templates.get(name)

    def get_for_count(self, image_count: int) -> Optional[LayoutTemplate]:
        """
        Get the best template for a given number of images.
        Returns the first template matching exactly, or the closest match.
        """
        if image_count in self._by_count:
            return self._by_count[image_count][0]

        # Find closest match (prefer fewer cells over more)
        available = sorted(self._by_count.keys())
        if not available:
            return None

        # Find the largest count that doesn't exceed image_count
        best = available[0]
        for c in available:
            if c <= image_count:
                best = c
            else:
                break
        return self._by_count[best][0]

    def get_templates_for_count(self, image_count: int) -> List[LayoutTemplate]:
        """Get ALL templates matching a specific image count."""
        return self._by_count.get(image_count, [])

    def all_templates(self) -> List[LayoutTemplate]:
        """Return all registered templates."""
        return list(self._templates.values())

    @property
    def template_count(self) -> int:
        return len(self._templates)
