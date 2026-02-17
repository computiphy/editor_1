import pytest
from pathlib import Path
from src.core.models import ImageScore

def test_image_score_initialization():
    """
    Rationale: Validates the primary domain model for quality scoring.
    Details: This test ensures that the `ImageScore` dataclass correctly stores all quality metrics 
    (blur, BRISQUE, etc.) which are essential for the downstream culling logic.
    """
    path = Path("test.jpg")
    score = ImageScore(
        path=path,
        blur_score=150.0,
        fft_energy=0.5,
        brisque_score=20.0,
        niqe_score=4.0,
        has_faces=True,
        blink_detected=False,
        expression_score=0.8,
        overall_quality=0.9,
        passed=True
    )
    assert score.path == path
    assert score.overall_quality == 0.9
    assert score.passed is True
