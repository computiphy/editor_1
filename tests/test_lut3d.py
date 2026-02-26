"""
Tests for P4: 3D LUT Engine (.cube format)
============================================
Validates .cube file parsing, tetrahedral interpolation,
and end-to-end LUT application.

.cube format reference: https://resolve.training/cube-lut-specification/
"""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def identity_cube_file():
    """Create a minimal 2x2x2 identity .cube LUT file."""
    content = """# Identity LUT
TITLE "Identity"
LUT_3D_SIZE 2

0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""
    fd, path = tempfile.mkstemp(suffix=".cube")
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    yield path
    os.unlink(path)


@pytest.fixture
def warm_cube_file():
    """Create a 2x2x2 .cube LUT that warms the image (boost red, reduce blue)."""
    content = """# Warm LUT
TITLE "Warm"
LUT_3D_SIZE 2

0.05 0.0 0.0
1.0 0.0 0.0
0.05 1.0 0.0
1.0 1.0 0.0
0.0 0.0 0.8
1.0 0.0 0.8
0.0 1.0 0.8
1.0 1.0 0.8
"""
    fd, path = tempfile.mkstemp(suffix=".cube")
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    yield path
    os.unlink(path)


@pytest.fixture
def gradient_image():
    """A simple gradient image for LUT testing."""
    rng = np.random.RandomState(42)
    return rng.rand(50, 50, 3).astype(np.float32)


# ── .cube File Parsing ─────────────────────────────────────────

class TestCubeParser:

    def test_parse_identity_cube(self, identity_cube_file):
        """Should parse a valid .cube file and extract the 3D LUT data."""
        from src.color.lut3d import parse_cube_file
        lut, size = parse_cube_file(identity_cube_file)
        assert size == 2
        assert lut.shape == (2, 2, 2, 3)

    def test_parse_extracts_correct_values(self, identity_cube_file):
        """Identity LUT corners should map inputs to themselves."""
        from src.color.lut3d import parse_cube_file
        lut, size = parse_cube_file(identity_cube_file)
        # (0,0,0) → (0,0,0)
        np.testing.assert_allclose(lut[0, 0, 0], [0, 0, 0], atol=0.01)
        # (1,1,1) → (1,1,1)
        np.testing.assert_allclose(lut[1, 1, 1], [1, 1, 1], atol=0.01)

    def test_parse_warm_lut(self, warm_cube_file):
        """Warm LUT should reduce blue at the high-blue corner."""
        from src.color.lut3d import parse_cube_file
        lut, size = parse_cube_file(warm_cube_file)
        # Corner (0,0,1) = pure blue input → should have reduced blue
        assert lut[0, 0, 1, 2] < 1.0, "Blue should be reduced in warm LUT"


# ── Tetrahedral Interpolation ──────────────────────────────────

class TestTetrahedralInterpolation:

    def test_identity_lut_is_passthrough(self, identity_cube_file, gradient_image):
        """An identity LUT should not change the image."""
        from src.color.lut3d import apply_lut3d
        result = apply_lut3d(gradient_image, identity_cube_file)
        max_error = np.abs(result - gradient_image).max()
        assert max_error < 0.05, f"Identity LUT should be passthrough, error={max_error:.4f}"

    def test_warm_lut_reduces_blue(self, warm_cube_file, gradient_image):
        """The warm LUT should reduce the blue channel."""
        from src.color.lut3d import apply_lut3d
        result = apply_lut3d(gradient_image, warm_cube_file)
        # Average blue should decrease
        assert result[:, :, 2].mean() < gradient_image[:, :, 2].mean(), \
            "Warm LUT should reduce blue"

    def test_output_shape_and_dtype(self, identity_cube_file, gradient_image):
        """Output should match input shape and be float32."""
        from src.color.lut3d import apply_lut3d
        result = apply_lut3d(gradient_image, identity_cube_file)
        assert result.shape == gradient_image.shape
        assert result.dtype == np.float32

    def test_output_range(self, warm_cube_file, gradient_image):
        """Output should be clamped to [0, 1]."""
        from src.color.lut3d import apply_lut3d
        result = apply_lut3d(gradient_image, warm_cube_file)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_no_banding_gradients(self, identity_cube_file):
        """
        Tetrahedral interpolation should produce smooth gradients
        without the banding artifacts of trilinear interpolation.
        """
        from src.color.lut3d import apply_lut3d
        # Create a perfect gradient
        grad = np.linspace(0, 1, 256).reshape(1, 256, 1).repeat(3, axis=2).astype(np.float32)
        grad = np.repeat(grad, 10, axis=0)  # 10x256x3
        result = apply_lut3d(grad, identity_cube_file)
        # Check adjacent pixel differences are smooth (no large jumps)
        diffs = np.abs(np.diff(result[5, :, 0]))
        max_jump = diffs.max()
        assert max_jump < 0.02, f"Gradient should be smooth, max_jump={max_jump:.4f}"
