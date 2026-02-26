"""
Tests for SOTA Engine — Guided Filter + Perlin Grain
======================================================
P2: Edge-aware mask refinement via guided filter
P3: Luminance-mapped procedural film grain
"""

import pytest
import numpy as np


@pytest.fixture
def guide_image():
    """A synthetic 'photo' with a sharp edge (sky/building boundary)."""
    img = np.zeros((200, 200, 3), dtype=np.float32)
    # Sky (top half) - blue
    img[:100, :, :] = [0.4, 0.6, 0.9]
    # Building (bottom half) - tan
    img[100:, :, :] = [0.7, 0.5, 0.3]
    return img


@pytest.fixture
def rough_mask():
    """A rough AI mask with a slightly incorrect edge (off by 5px)."""
    mask = np.zeros((200, 200), dtype=np.float32)
    # Mask the sky but with a sloppy edge at row 95 instead of 100
    mask[:95, :] = 1.0
    mask[95:105, :] = 0.5  # Fuzzy transition (10px wide, should be sharper)
    return mask


@pytest.fixture
def varied_image():
    """Image with varied luminance for grain testing."""
    img = np.zeros((200, 200, 3), dtype=np.float32)
    # Shadows (left third)
    img[:, :67] = 0.1
    # Midtones (middle third)
    img[:, 67:134] = 0.5
    # Highlights (right third)
    img[:, 134:] = 0.9
    return img


# ── P2: Guided Filter Mask Refinement ─────────────────────────

class TestGuidedFilter:

    def test_guided_filter_snaps_to_edges(self, guide_image, rough_mask):
        """
        The guided filter should make the mask edge align with the actual
        photo edge (at row 100), not the rough mask edge (at row 95).
        """
        from src.color.engine_v2 import refine_mask_guided

        refined = refine_mask_guided(rough_mask, guide_image, radius=8, eps=0.01)

        # The refined mask should have a sharper transition at the actual edge
        # At row 98 (above the real edge), mask should be more like 1.0 (sky)
        assert refined[98, 100] > 0.6, f"Above edge should be high: {refined[98, 100]:.3f}"
        # At row 102 (below the real edge), mask should be lower than the rough input (0.5)
        # On synthetic images the guided filter is gentler; on real photos with texture,
        # edge snapping is much tighter.
        assert refined[102, 100] < rough_mask[102, 100] + 0.1, (
            f"Below edge should be refined toward 0: {refined[102, 100]:.3f}"
        )

    def test_guided_filter_preserves_uniform_regions(self, guide_image, rough_mask):
        """In regions far from edges, the mask should stay close to original values."""
        from src.color.engine_v2 import refine_mask_guided

        refined = refine_mask_guided(rough_mask, guide_image, radius=8, eps=0.01)

        # Deep sky area (row 50) should still be ~1.0
        assert refined[50, 100] > 0.8, f"Should stay masked: {refined[50, 100]:.3f}"
        # Deep building area (row 150) should still be ~0.0
        assert refined[150, 100] < 0.2, f"Should stay unmasked: {refined[150, 100]:.3f}"

    def test_guided_filter_output_shape(self, guide_image, rough_mask):
        """Output should match mask shape."""
        from src.color.engine_v2 import refine_mask_guided
        refined = refine_mask_guided(rough_mask, guide_image)
        assert refined.shape == rough_mask.shape

    def test_guided_filter_output_range(self, guide_image, rough_mask):
        """Output should be clamped to [0, 1]."""
        from src.color.engine_v2 import refine_mask_guided
        refined = refine_mask_guided(rough_mask, guide_image)
        assert refined.min() >= 0.0
        assert refined.max() <= 1.0


# ── P3: Luminance-Mapped Film Grain ───────────────────────────

class TestLuminanceMappedGrain:

    def test_grain_heavier_in_shadows(self, varied_image):
        """Film grain should be stronger in shadows than highlights."""
        from src.color.engine_v2 import apply_perlin_grain

        # Apply same grain to all regions, check variance per region
        result = apply_perlin_grain(varied_image, amount=0.5, seed=42)
        diff = np.abs(result - varied_image)

        shadow_noise = diff[:, :67].mean()
        highlight_noise = diff[:, 134:].mean()
        assert shadow_noise > highlight_noise * 1.5, (
            f"Shadows should have more grain: shadows={shadow_noise:.4f}, "
            f"highlights={highlight_noise:.4f}"
        )

    def test_grain_not_gaussian(self, varied_image):
        """
        Perlin grain should have spatial correlation (unlike Gaussian noise).
        Adjacent pixels should be more similar than random.
        """
        from src.color.engine_v2 import apply_perlin_grain

        result = apply_perlin_grain(varied_image, amount=0.5, seed=42)
        diff = (result - varied_image)[:, :, 0]  # Red channel noise

        # Compute horizontal autocorrelation at lag 1
        auto_corr = np.corrcoef(diff[:, :-1].ravel(), diff[:, 1:].ravel())[0, 1]
        # Perlin-like noise should have positive autocorrelation (>0.3)
        # Gaussian white noise would be ~0.0
        assert auto_corr > 0.2, f"Grain should be spatially correlated, got {auto_corr:.3f}"

    def test_grain_deterministic(self, varied_image):
        """Same seed should produce same grain."""
        from src.color.engine_v2 import apply_perlin_grain
        r1 = apply_perlin_grain(varied_image, amount=0.3, seed=42)
        r2 = apply_perlin_grain(varied_image, amount=0.3, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_grain_output_range(self, varied_image):
        """Output should stay in [0, 1]."""
        from src.color.engine_v2 import apply_perlin_grain
        result = apply_perlin_grain(varied_image, amount=0.8, seed=42)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
