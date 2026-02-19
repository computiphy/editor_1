"""
Tests for AI Layout Generator (Phase 3 — Rule-Based).
"""
import pytest
from src.layout.ai_generator import AILayoutGenerator, LayoutStyle


# ── Core Generation Tests ──────────────────────────────────────

def test_generates_correct_cell_count():
    """Rationale: Must produce exactly the right number of cells."""
    gen = AILayoutGenerator(seed=42)
    for count in [1, 2, 3, 4, 5, 6, 8, 10]:
        layout = gen.generate(image_count=count, style="classic")
        assert len(layout.cells) == count, \
            f"Expected {count} cells, got {len(layout.cells)}"


def test_cells_within_bounds():
    """Rationale: All cells must be within [0, 1] normalized space."""
    gen = AILayoutGenerator(seed=42)
    for style in ["classic", "elegant", "minimal", "dynamic", "magazine"]:
        for count in [1, 2, 3, 4, 5, 6]:
            layout = gen.generate(image_count=count, style=style)
            for cell in layout.cells:
                assert 0.0 <= cell.x < 1.0, f"{style}/{count}: x={cell.x}"
                assert 0.0 <= cell.y < 1.0, f"{style}/{count}: y={cell.y}"
                assert cell.w > 0, f"{style}/{count}: w={cell.w}"
                assert cell.h > 0, f"{style}/{count}: h={cell.h}"
                assert cell.x + cell.w <= 1.01, f"{style}/{count}: x+w={cell.x + cell.w}"
                assert cell.y + cell.h <= 1.01, f"{style}/{count}: y+h={cell.y + cell.h}"


def test_no_overlapping_cells():
    """Rationale: Cells should not overlap (would cause visual artifacts)."""
    gen = AILayoutGenerator(seed=42)
    for style in ["classic", "minimal"]:
        for count in [2, 3, 4]:
            layout = gen.generate(image_count=count, style=style)
            for i in range(len(layout.cells)):
                for j in range(i + 1, len(layout.cells)):
                    a, b = layout.cells[i], layout.cells[j]
                    overlap = not (
                        a.x + a.w <= b.x + 0.01 or
                        b.x + b.w <= a.x + 0.01 or
                        a.y + a.h <= b.y + 0.01 or
                        b.y + b.h <= a.y + 0.01
                    )
                    assert not overlap, \
                        f"{style}/{count}: Cells {i} and {j} overlap"


# ── Reproducibility Tests ──────────────────────────────────────

def test_seed_reproducibility():
    """Rationale: Same seed must produce identical layout."""
    gen1 = AILayoutGenerator(seed=123)
    gen2 = AILayoutGenerator(seed=123)

    l1 = gen1.generate(image_count=4, style="classic")
    l2 = gen2.generate(image_count=4, style="classic")

    assert len(l1.cells) == len(l2.cells)
    for c1, c2 in zip(l1.cells, l2.cells):
        assert abs(c1.x - c2.x) < 0.001
        assert abs(c1.y - c2.y) < 0.001
        assert abs(c1.w - c2.w) < 0.001
        assert abs(c1.h - c2.h) < 0.001


def test_different_seeds_differ():
    """Rationale: Different seeds should produce different layouts."""
    gen1 = AILayoutGenerator(seed=1)
    gen2 = AILayoutGenerator(seed=999)

    l1 = gen1.generate(image_count=4, style="classic")
    l2 = gen2.generate(image_count=4, style="classic")

    # At least one cell should differ
    diffs = sum(
        abs(c1.x - c2.x) + abs(c1.y - c2.y) + abs(c1.w - c2.w) + abs(c1.h - c2.h)
        for c1, c2 in zip(l1.cells, l2.cells)
    )
    assert diffs > 0.001, "Different seeds should produce different layouts"


# ── Style Tests ────────────────────────────────────────────────

def test_elegant_has_more_whitespace():
    """Rationale: Elegant style should use less total area than classic."""
    gen = AILayoutGenerator(seed=42)
    classic = gen.generate(image_count=3, style="classic")
    elegant = gen.generate(image_count=3, style="elegant")

    area_classic = sum(c.w * c.h for c in classic.cells)
    area_elegant = sum(c.w * c.h for c in elegant.cells)

    assert area_elegant < area_classic, \
        "Elegant should have more whitespace (less total cell area)"


def test_all_styles_produce_valid_layouts():
    """Rationale: Every style must produce valid layouts for all counts."""
    gen = AILayoutGenerator(seed=42)
    for style in ["classic", "elegant", "minimal", "dynamic", "magazine"]:
        for count in [1, 2, 3, 4, 5, 6, 8]:
            layout = gen.generate(image_count=count, style=style)
            assert len(layout.cells) == count
            assert layout.score > 0, f"{style}/{count}: score={layout.score}"
            assert layout.style == style


# ── Scoring Tests ──────────────────────────────────────────────

def test_variants_scored_differently():
    """Rationale: Multiple variants should have different scores."""
    gen = AILayoutGenerator(seed=42)
    variants = gen.generate_all_variants(
        image_count=4, style="classic", num_variants=5
    )
    scores = [v.score for v in variants]
    assert len(set(scores)) > 1, "Variants should have varied scores"
    # Best variant should be first (sorted descending)
    assert scores[0] >= scores[-1]


def test_best_variant_selected():
    """Rationale: generate() should return the highest-scoring variant."""
    gen = AILayoutGenerator(seed=42)
    best = gen.generate(image_count=3, style="classic", num_variants=5)
    all_variants = gen.generate_all_variants(
        image_count=3, style="classic", num_variants=5
    )
    assert best.score == all_variants[0].score, \
        "generate() should return the best variant"


# ── Aspect Ratio Tests ─────────────────────────────────────────

def test_respects_aspect_ratios():
    """Rationale: AI should consider aspect ratios in layout generation."""
    gen = AILayoutGenerator(seed=42)

    # All portrait images
    portrait = gen.generate(
        image_count=2,
        aspect_ratios=[0.67, 0.67],  # Portrait
        style="classic"
    )

    # All landscape images
    landscape = gen.generate(
        image_count=2,
        aspect_ratios=[1.5, 1.5],  # Landscape
        style="classic"
    )

    # Both should produce valid layouts
    assert len(portrait.cells) == 2
    assert len(landscape.cells) == 2
    assert portrait.score > 0
    assert landscape.score > 0


# ── Integration with Engine ────────────────────────────────────

def test_engine_ai_mode():
    """Rationale: Engine with mode='ai' should use AI generator."""
    from src.layout.engine import AlbumLayoutEngine
    from pathlib import Path
    from PIL import Image
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        final_dir = tmpdir / "final"
        final_dir.mkdir()
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Create 4 test images
        for i in range(4):
            img = Image.new("RGB", (400, 300), (100 + i * 30, 80, 150))
            img.save(str(final_dir / f"img_{i:03d}.jpg"))

        engine = AlbumLayoutEngine(
            mode="ai",
            page_size=(800, 600),
            images_per_page=4,
            ai_style="magazine",
            ai_seed=42,
        )

        project = engine.generate_album(
            final_dir=final_dir,
            cutouts_dir=None,
            output_dir=output_dir,
        )

        assert project.total_images == 4
        assert len(project.pages) == 1
        assert project.pages[0].layout_mode == "ai"
        assert (output_dir / "album" / "page_001.jpg").exists()
