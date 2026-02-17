import pytest
import numpy as np
from src.culling.face_analyzer import FaceAnalyzer

def test_face_analyzer_detects_no_faces():
    """
    Rationale: Verifies detection negative case.
    Details: This test ensures that the `FaceAnalyzer` correctly identifies images without faces, 
    preventing false positives in the blink detection and expression scoring modules.
    """
    # Constant image has no faces
    no_face_img = np.zeros((300, 300, 3), dtype=np.uint8)
    analyzer = FaceAnalyzer()
    
    assert analyzer.has_faces(no_face_img) is False

def test_face_analyzer_detects_blink_mock(mocker):
    """
    Rationale: Validates EAR-based blink detection logic.
    Details: This test uses mocking to verify the logic branch that determines if a subject is 
    blinking based on the Eye Aspect Ratio (EAR), which is critical for culling "bad" shots.
    """
    # We'll mock the mesh result for blink detection test
    analyzer = FaceAnalyzer()
    
    # Mocking media pipe result
    mock_mesh = mocker.Mock()
    # Mock landmarks for eyes (e.g. eye aspect ratio)
    # We will just verify the logic branch for now
    mocker.patch.object(analyzer, "_get_eye_aspect_ratio", return_value=0.15) # Blinking
    assert analyzer.is_blinking(None) is True
    
    mocker.patch.object(analyzer, "_get_eye_aspect_ratio", return_value=0.3) # Open
    assert analyzer.is_blinking(None) is False
