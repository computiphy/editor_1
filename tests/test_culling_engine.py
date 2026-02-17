import pytest
from pathlib import Path
import numpy as np
from src.culling.engine import CullingEngine
from src.culling.blur_detector import BlurDetector
from src.culling.face_analyzer import FaceAnalyzer
from src.culling.quality_assessor import QualityAssessor
from src.culling.duplicate_cluster import DuplicateClusterer

@pytest.fixture
def mock_engines(mocker):
    return {
        "blur": mocker.Mock(spec=BlurDetector),
        "face": mocker.Mock(spec=FaceAnalyzer),
        "quality": mocker.Mock(spec=QualityAssessor),
        "clusterer": mocker.Mock(spec=DuplicateClusterer)
    }

def test_culling_engine_scores_image(mock_engines, mocker):
    """
    Rationale: Ensures the composite culling engine correctly aggregates scores from sub-engines.
    Details: This test validates the "Brain" of the culling stage, ensuring that blur, quality, 
    and face data are combined into a single `ImageScore` object.
    """
    # Setup mocks
    mock_engines["blur"].calculate_variance.return_value = 150.0
    mock_engines["blur"].is_blurry.return_value = False
    mock_engines["face"].has_faces.return_value = True
    mock_engines["face"].is_blinking.return_value = False
    mock_engines["quality"].calculate_scores.return_value = (20.0, 4.0)
    
    engine = CullingEngine(
        blur_detector=mock_engines["blur"],
        face_analyzer=mock_engines["face"],
        quality_assessor=mock_engines["quality"],
        clusterer=mock_engines["clusterer"]
    )
    
    # Mock image loading at the site of use
    mocker.patch("src.culling.engine.load_image", return_value=np.zeros((100, 100, 3)))
    
    score = engine.evaluate_image(Path("test.jpg"))
    
    assert score.blur_score == 150.0
    assert score.has_faces is True
    assert score.passed is True
