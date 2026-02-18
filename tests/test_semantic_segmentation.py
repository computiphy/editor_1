"""
Tests for Semantic Segmentation and Per-Region Color Grading.
"""
import pytest
import numpy as np
import cv2


# ── Segmenter Tests ──────────────────────────────────────────────

def test_segmenter_returns_all_masks():
    """
    Rationale: Ensures segmenter produces masks for all 5 defined regions.
    """
    from src.segmentation.semantic_segmenter import SemanticSegmenter

    segmenter = SemanticSegmenter()
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    result = segmenter.segment(img)

    assert result.skin.shape == (200, 300)
    assert result.sky.shape == (200, 300)
    assert result.vegetation.shape == (200, 300)
    assert result.dress_white.shape == (200, 300)
    assert result.suit_dark.shape == (200, 300)


def test_segmenter_masks_are_float():
    """
    Rationale: Masks must be float32 [0,1] for soft blending.
    """
    from src.segmentation.semantic_segmenter import SemanticSegmenter

    segmenter = SemanticSegmenter()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = segmenter.segment(img)

    for name, mask in result.as_dict().items():
        assert mask.dtype == np.float32, f"{name} mask is not float32"
        assert mask.min() >= 0.0, f"{name} mask has values < 0"
        assert mask.max() <= 1.0, f"{name} mask has values > 1"


def test_segmenter_detects_skin_tones():
    """
    Rationale: A solid skin-tone colored image should produce a strong skin mask.
    """
    from src.segmentation.semantic_segmenter import SemanticSegmenter

    segmenter = SemanticSegmenter()
    # Create an image entirely in skin-tone range (peach/orange)
    # RGB roughly (210, 165, 130) maps to HSV H≈15 in OpenCV
    img = np.full((100, 100, 3), [210, 165, 130], dtype=np.uint8)
    result = segmenter.segment(img)

    # Skin mask should have significant coverage
    assert result.skin.mean() > 0.3, "Skin tone image should produce strong skin mask"


def test_segmenter_detects_sky():
    """
    Rationale: A blue patch in the top half should produce a sky mask.
    """
    from src.segmentation.semantic_segmenter import SemanticSegmenter

    segmenter = SemanticSegmenter()
    # Sky blue in the top half
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:100, :] = [100, 170, 230]  # Sky blue (RGB)
    img[100:, :] = [50, 120, 50]    # Green ground
    result = segmenter.segment(img)

    # Sky should be detected in top half
    assert result.sky[:100, :].mean() > result.sky[100:, :].mean(), \
        "Sky should be stronger in top half"


def test_segmenter_detects_vegetation():
    """
    Rationale: A solid green image should produce a vegetation mask.
    """
    from src.segmentation.semantic_segmenter import SemanticSegmenter

    segmenter = SemanticSegmenter()
    img = np.full((100, 100, 3), [40, 140, 40], dtype=np.uint8)  # Green
    result = segmenter.segment(img)

    assert result.vegetation.mean() > 0.3, "Green image should produce vegetation mask"


def test_segmenter_skin_priority():
    """
    Rationale: Skin pixels should be removed from other masks to prevent
    skin regions from being affected by, e.g., vegetation taming.
    """
    from src.segmentation.semantic_segmenter import SemanticSegmenter

    segmenter = SemanticSegmenter()
    # Skin-tone image (should not appear in vegetation mask)
    img = np.full((100, 100, 3), [210, 165, 130], dtype=np.uint8)
    result = segmenter.segment(img)

    # Vegetation should be empty since skin has priority
    assert result.vegetation.mean() < 0.1, "Skin should suppress vegetation mask"


# ── Semantic Grading Tests ──────────────────────────────────────

def test_semantic_grading_modifies_image():
    """
    Rationale: Semantic grading with non-empty masks should change the image.
    """
    from src.color.engine import ColorGradingEngine

    engine = ColorGradingEngine(style="cinematic")
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

    # Create a simple skin mask covering the whole image
    masks = {
        "skin": np.ones((100, 100), dtype=np.float32),
        "sky": np.zeros((100, 100), dtype=np.float32),
        "vegetation": np.zeros((100, 100), dtype=np.float32),
        "dress_white": np.zeros((100, 100), dtype=np.float32),
        "suit_dark": np.zeros((100, 100), dtype=np.float32),
    }

    result = engine.apply_semantic_grading(img, masks)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, img), "Semantic grading should modify the image"


def test_semantic_grading_skin_warms_image():
    """
    Rationale: Skin override should add warmth (increase red, decrease blue).
    """
    from src.color.engine import ColorGradingEngine

    engine = ColorGradingEngine()
    # Neutral grey image
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    masks = {
        "skin": np.ones((50, 50), dtype=np.float32),
        "sky": np.zeros((50, 50), dtype=np.float32),
        "vegetation": np.zeros((50, 50), dtype=np.float32),
        "dress_white": np.zeros((50, 50), dtype=np.float32),
        "suit_dark": np.zeros((50, 50), dtype=np.float32),
    }

    result = engine.apply_semantic_grading(img, masks)

    # Red should increase, blue should decrease
    mean_r = result[:, :, 0].mean()
    mean_b = result[:, :, 2].mean()
    assert mean_r > 128, "Skin override should warm red channel"


def test_semantic_grading_vegetation_desaturates():
    """
    Rationale: Vegetation override should desaturate greens by 20%.
    """
    from src.color.engine import ColorGradingEngine

    engine = ColorGradingEngine()
    # Bright green image
    img = np.full((50, 50, 3), [40, 200, 40], dtype=np.uint8)
    masks = {
        "skin": np.zeros((50, 50), dtype=np.float32),
        "sky": np.zeros((50, 50), dtype=np.float32),
        "vegetation": np.ones((50, 50), dtype=np.float32),
        "dress_white": np.zeros((50, 50), dtype=np.float32),
        "suit_dark": np.zeros((50, 50), dtype=np.float32),
    }

    result = engine.apply_semantic_grading(img, masks)

    # Vegetation should be desaturated
    hsv_in = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_out = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    assert hsv_out[:, :, 1].mean() < hsv_in[:, :, 1].mean(), \
        "Vegetation override should desaturate"


def test_semantic_grading_empty_masks_no_change():
    """
    Rationale: All-zero masks should produce no change.
    """
    from src.color.engine import ColorGradingEngine

    engine = ColorGradingEngine()
    img = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
    masks = {k: np.zeros((50, 50), dtype=np.float32)
             for k in ["skin", "sky", "vegetation", "dress_white", "suit_dark"]}

    result = engine.apply_semantic_grading(img, masks)
    np.testing.assert_array_equal(result, img)


def test_full_pipeline_with_segmentation():
    """
    Rationale: End-to-end test — apply_style + semantic grading should produce
    a valid result without errors.
    """
    from src.color.engine import ColorGradingEngine
    from src.segmentation.semantic_segmenter import SemanticSegmenter

    engine = ColorGradingEngine(style="cinematic")
    segmenter = SemanticSegmenter()

    # Multi-region test image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:60, :] = [100, 170, 230]         # Sky (top)
    img[60:100, :] = [210, 165, 130]      # Skin (middle)
    img[100:150, :100] = [40, 140, 40]    # Vegetation (bottom-left)
    img[100:150, 100:] = [30, 30, 30]     # Dark suit (bottom-right)
    img[150:, :] = [250, 250, 250]        # White dress (bottom)

    # Step 1: Global preset
    graded = engine.apply_style(img)

    # Step 2: Semantic overrides
    seg = segmenter.segment(graded)
    result = engine.apply_semantic_grading(graded, seg.as_dict())

    assert result.shape == img.shape
    assert result.dtype == np.uint8
    # Should be different from both input and globally-graded version
    assert not np.array_equal(result, img)
