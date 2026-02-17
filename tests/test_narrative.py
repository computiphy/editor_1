import pytest
import torch
import numpy as np
from src.narrative.engine import NarrativeEngine

def test_narrative_engine_clustering(mocker):
    """
    Rationale: Validates that the narrative engine can group images into chapters.
    Details: This test verifies that visual embeddings (CLIP) and timestamps can be used 
    to create a storyline, which is the foundation of the album layout.
    """
    engine = NarrativeEngine()
    
    # Mock CLIP embeddings
    # 5 images, 2 clearly different scenes (3 from scene A, 2 from scene B)
    embeddings = np.zeros((5, 512))
    embeddings[0:3, 0] = 1.0 # Scene A
    embeddings[3:5, 1] = 1.0 # Scene B
    
    mocker.patch.object(engine, "_compute_embeddings", return_value=embeddings)
    
    paths = [f"img{i}.jpg" for i in range(5)]
    chapters = engine.group_to_chapters(paths)
    
    assert len(chapters) == 2
    assert chapters[0].name == "Chapter 1"
    assert len(chapters[0].images) == 3
    assert len(chapters[1].images) == 2
