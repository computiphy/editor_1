"""
Tests for Album Layout Engine (Stage 9)
Tests template registry, color analysis, background selection,
rendering, and project state.
"""
import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from PIL import Image


# ── Template Registry Tests ────────────────────────────────────

def test_template_registry_loads_bundled():
    """Rationale: Bundled templates must always be available."""
    from src.layout.template_registry import TemplateRegistry

    registry = TemplateRegistry()
    assert registry.template_count >= 10, "Should have at least 10 bundled templates"


def test_template_registry_lookup_by_count():
    """Rationale: Must find templates for 1–6 image counts."""
    from src.layout.template_registry import TemplateRegistry

    registry = TemplateRegistry()
    for count in [1, 2, 3, 4, 5, 6]:
        template = registry.get_for_count(count)
        assert template is not None, f"No template for {count} images"
        assert template.image_count == count, \
            f"Template for {count} images has {template.image_count} cells"


def test_template_cells_normalized():
    """Rationale: All cell coordinates must be in [0, 1] range."""
    from src.layout.template_registry import TemplateRegistry

    registry = TemplateRegistry()
    for template in registry.all_templates():
        for cell in template.cells:
            assert 0.0 <= cell.x <= 1.0, f"{template.name}: x={cell.x} out of range"
            assert 0.0 <= cell.y <= 1.0, f"{template.name}: y={cell.y} out of range"
            assert 0.0 < cell.w <= 1.0, f"{template.name}: w={cell.w} out of range"
            assert 0.0 < cell.h <= 1.0, f"{template.name}: h={cell.h} out of range"


def test_template_from_json_file():
    """Rationale: User-provided JSON templates must load correctly."""
    from src.layout.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        template_file = Path(tmpdir) / "custom.json"
        template_file.write_text(json.dumps({
            "name": "custom_test",
            "description": "Test template",
            "image_count": 2,
            "cells": [
                {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},
                {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0},
            ]
        }))

        registry = TemplateRegistry(user_template_dir=Path(tmpdir))
        custom = registry.get_by_name("custom_test")
        assert custom is not None
        assert custom.image_count == 2


# ── Color Analyzer Tests ───────────────────────────────────────

def test_dominant_color_extraction():
    """Rationale: K-Means must extract the majority color from a solid image."""
    from src.layout.color_analyzer import extract_dominant_color

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a solid red image
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        path = Path(tmpdir) / "red.jpg"
        img.save(str(path), "JPEG")

        lab = extract_dominant_color(path)
        assert len(lab) == 3
        # Red in LAB: L≈53, A≈80+, B≈67+
        assert lab[1] > 128, "Red should have high A value in LAB"


def test_delta_e_same_color():
    """Rationale: Same color should have ΔE = 0."""
    from src.layout.color_analyzer import delta_e

    color = (50.0, 128.0, 128.0)
    assert delta_e(color, color) == 0.0


def test_delta_e_different_colors():
    """Rationale: Very different colors should have high ΔE."""
    from src.layout.color_analyzer import delta_e

    white_lab = (255.0, 128.0, 128.0)
    black_lab = (0.0, 128.0, 128.0)
    assert delta_e(white_lab, black_lab) > 200


# ── Background Selector Tests ─────────────────────────────────

def test_background_selector_empty_dir():
    """Rationale: No candidates → return None gracefully."""
    from src.layout.background_selector import BackgroundSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        selector = BackgroundSelector(Path(tmpdir))
        assert selector.candidate_count == 0
        result = selector.select([Path("some_image.jpg")])
        assert result is None


def test_background_selector_picks_closest():
    """Rationale: Selector must pick the background with minimum ΔE."""
    from src.layout.background_selector import BackgroundSelector

    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir) / "bgs"
        img_dir = Path(tmpdir) / "imgs"
        bg_dir.mkdir()
        img_dir.mkdir()

        # Create two backgrounds: one red, one blue
        Image.new("RGB", (100, 100), (255, 0, 0)).save(str(bg_dir / "red.jpg"))
        Image.new("RGB", (100, 100), (0, 0, 255)).save(str(bg_dir / "blue.jpg"))

        # Create a red test image
        Image.new("RGB", (100, 100), (200, 20, 20)).save(str(img_dir / "test.jpg"))

        selector = BackgroundSelector(bg_dir)
        result = selector.select([img_dir / "test.jpg"])

        assert result is not None
        assert "red" in result.name.lower(), "Should pick red background for red image"


# ── Renderer Tests ─────────────────────────────────────────────

def test_renderer_creates_jpeg():
    """Rationale: Renderer must produce a valid JPEG file."""
    from src.layout.renderer import AlbumRenderer
    from src.layout.models import PageLayout, CellPlacement

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        img = Image.new("RGB", (200, 200), (100, 150, 200))
        img_path = tmpdir / "test.jpg"
        img.save(str(img_path))

        page = PageLayout(
            page_number=1,
            cells=[CellPlacement(
                image_path=img_path, cutout_path=None,
                x=60, y=60, width=500, height=400, z_order=0,
            )],
        )

        renderer = AlbumRenderer(page_width=800, page_height=600)
        out_path = tmpdir / "page_001.jpg"
        renderer.render_page(page, out_path)

        assert out_path.exists()
        rendered = Image.open(out_path)
        assert rendered.size == (800, 600)
        rendered.close()


def test_renderer_with_background():
    """Rationale: Background image must fill the canvas."""
    from src.layout.renderer import AlbumRenderer
    from src.layout.models import PageLayout, CellPlacement

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create background
        bg = Image.new("RGB", (300, 200), (50, 50, 50))
        bg_path = tmpdir / "bg.jpg"
        bg.save(str(bg_path))

        # Create test image
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        img_path = tmpdir / "photo.jpg"
        img.save(str(img_path))

        page = PageLayout(
            page_number=1,
            cells=[CellPlacement(
                image_path=img_path, cutout_path=None,
                x=50, y=50, width=200, height=150, z_order=0,
            )],
            background_path=bg_path,
        )

        renderer = AlbumRenderer(page_width=400, page_height=300)
        out_path = tmpdir / "page_bg.jpg"
        renderer.render_page(page, out_path)

        assert out_path.exists()


# ── Project State Tests ────────────────────────────────────────

def test_project_serialization_roundtrip():
    """Rationale: project.json must survive save/load without data loss."""
    from src.layout.models import AlbumProject, PageLayout, CellPlacement

    with tempfile.TemporaryDirectory() as tmpdir:
        project = AlbumProject(
            config_snapshot={"mode": "template"},
            pages=[PageLayout(
                page_number=1,
                cells=[CellPlacement(
                    image_path=Path("final/img.jpg"), cutout_path=None,
                    x=60, y=60, width=500, height=400,
                )],
                layout_mode="template",
                template_name="hero_right_2",
            )],
            total_images=10,
        )

        save_path = Path(tmpdir) / "project.json"
        project.save(save_path)

        loaded = AlbumProject.load(save_path)
        assert loaded.total_images == 10
        assert len(loaded.pages) == 1
        assert loaded.pages[0].template_name == "hero_right_2"
        assert loaded.pages[0].cells[0].width == 500


# ── Engine Integration Tests ───────────────────────────────────

def test_engine_generates_album():
    """Rationale: End-to-end album generation from final/ directory."""
    from src.layout.engine import AlbumLayoutEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        final_dir = tmpdir / "final"
        final_dir.mkdir()
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Create 6 test images
        for i in range(6):
            img = Image.new("RGB", (400, 300), (100 + i * 20, 80, 150))
            img.save(str(final_dir / f"img_{i:03d}.jpg"))

        engine = AlbumLayoutEngine(
            page_size=(800, 600),
            images_per_page=3,
        )

        project = engine.generate_album(
            final_dir=final_dir,
            cutouts_dir=None,
            output_dir=output_dir,
        )

        assert project.total_images == 6
        assert len(project.pages) == 2  # 6 images / 3 per page

        album_dir = output_dir / "album"
        assert album_dir.exists()
        assert (album_dir / "page_001.jpg").exists()
        assert (album_dir / "page_002.jpg").exists()
        assert (album_dir / "project.json").exists()


def test_engine_auto_images_per_page():
    """Rationale: Auto mode should compute reasonable images per page."""
    from src.layout.engine import AlbumLayoutEngine

    engine = AlbumLayoutEngine()

    assert engine._compute_images_per_page(3) <= 2
    assert engine._compute_images_per_page(15) == 3
    assert engine._compute_images_per_page(30) == 4
    assert engine._compute_images_per_page(100) == 5
