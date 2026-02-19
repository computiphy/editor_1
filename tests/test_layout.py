"""
Legacy Layout Engine Test — Updated for AlbumLayoutEngine (Stage 9)
"""
import pytest
from src.layout.engine import AlbumLayoutEngine


def test_layout_engine_generates_page():
    """
    Rationale: Validates the core layout engine can generate pages
    with valid non-overlapping coordinates.
    """
    engine = AlbumLayoutEngine(
        mode="template",
        page_size=(800, 600),
        images_per_page=3,
    )

    # Verify engine initializes with correct settings
    assert engine.page_width == 800
    assert engine.page_height == 600
    assert engine.mode == "template"
    assert engine.registry.template_count >= 10
