"""
Album Layout Data Models
========================
Data classes for page layouts, cell placements, and album projects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime


@dataclass
class CellPlacement:
    """A single image placement within a page."""
    image_path: Path
    cutout_path: Optional[Path]
    x: int                          # Pixel X
    y: int                          # Pixel Y
    width: int                      # Cell width (px)
    height: int                     # Cell height (px)
    z_order: int = 0

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "cutout_path": str(self.cutout_path) if self.cutout_path else None,
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "z_order": self.z_order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CellPlacement":
        return cls(
            image_path=Path(d["image_path"]),
            cutout_path=Path(d["cutout_path"]) if d.get("cutout_path") else None,
            x=d["x"], y=d["y"],
            width=d["width"], height=d["height"],
            z_order=d.get("z_order", 0),
        )


@dataclass
class PageLayout:
    """Complete specification for a single album page."""
    page_number: int
    cells: List[CellPlacement]
    background_path: Optional[Path] = None
    background_color: Tuple[int, int, int] = (255, 255, 255)
    layout_mode: str = "template"
    template_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "cells": [c.to_dict() for c in self.cells],
            "background_path": str(self.background_path) if self.background_path else None,
            "background_color": list(self.background_color),
            "layout_mode": self.layout_mode,
            "template_name": self.template_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageLayout":
        return cls(
            page_number=d["page_number"],
            cells=[CellPlacement.from_dict(c) for c in d["cells"]],
            background_path=Path(d["background_path"]) if d.get("background_path") else None,
            background_color=tuple(d.get("background_color", [255, 255, 255])),
            layout_mode=d.get("layout_mode", "template"),
            template_name=d.get("template_name"),
        )


@dataclass
class AlbumProject:
    """Full album project state — serializable to project.json."""
    config_snapshot: dict
    pages: List[PageLayout]
    total_images: int
    generation_timestamp: str = ""

    def __post_init__(self):
        if not self.generation_timestamp:
            self.generation_timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "config_snapshot": self.config_snapshot,
            "pages": [p.to_dict() for p in self.pages],
            "total_images": self.total_images,
            "generation_timestamp": self.generation_timestamp,
        }

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "AlbumProject":
        with open(path) as f:
            d = json.load(f)
        return cls(
            config_snapshot=d["config_snapshot"],
            pages=[PageLayout.from_dict(p) for p in d["pages"]],
            total_images=d["total_images"],
            generation_timestamp=d.get("generation_timestamp", ""),
        )
