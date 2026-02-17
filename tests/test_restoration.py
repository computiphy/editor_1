import pytest
import torch
import numpy as np
from pathlib import Path
from src.restoration.engine import RestorationEngine
from src.core.enums import RestorationMode

@pytest.fixture
def dummy_image():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

def test_restoration_engine_initialization():
    """
    Rationale: Ensures the RestorationEngine can be initialized with default backends.
    Details: Validates that the engine correctly identifies the GPU/CPU backend for NAFNet and GFPGAN.
    """
    engine = RestorationEngine(backend="cpu")
    assert engine.backend == "cpu"
    assert engine.nafnet is None # Lazy loading

def test_restoration_routing_logic(dummy_image, mocker):
    """
    Rationale: Validates the auto-routing logic based on quality scores.
    Details: Ensures that the engine selects the correct restoration model (NAFNet vs GFPGAN) 
    depending on whether faces are detected or blur is present.
    """
    engine = RestorationEngine(backend="cpu")
    
    # Mock the sub-restorers
    mock_nafnet = mocker.patch("src.restoration.nafnet_restore.NAFNetRestorer")
    mock_gfpgan = mocker.patch("src.restoration.gfpgan_restore.GFPGANRestorer")
    
    engine.nafnet = mock_nafnet.return_value
    engine.gfpgan = mock_gfpgan.return_value
    
    # Case 1: Blurry image with faces -> GFPGAN + NAFNet
    result = engine.restore(dummy_image, has_faces=True, blur_score=50.0)
    assert engine.gfpgan.restore.called
    assert engine.nafnet.restore.called
    
    # Case 2: Blurry image no faces -> Just NAFNet
    engine.gfpgan.restore.reset_mock()
    engine.nafnet.restore.reset_mock()
    result = engine.restore(dummy_image, has_faces=False, blur_score=50.0)
    assert not engine.gfpgan.restore.called
    assert engine.nafnet.restore.called
