import pytest
import numpy as np
from PIL import Image
from src.culling.duplicate_cluster import DuplicateClusterer

def test_duplicate_clusterer_groups_same_images(tmp_path):
    """
    Rationale: Validates pHash clustering for identical or near-identical files.
    Details: This test ensures that the culling engine can identify duplicates to avoid 
    redundancy in the final album layout.
    """
    # Create two identical images
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"
    Image.fromarray(data).save(img1_path)
    Image.fromarray(data).save(img2_path)
    
    clusterer = DuplicateClusterer(threshold=5)
    clusters = clusterer.cluster_duplicates([str(img1_path), str(img2_path)])
    
    assert len(clusters) == 1
    assert len(clusters[0].images) == 2

def test_duplicate_clusterer_separates_different_images(tmp_path):
    """
    Rationale: Ensures distinct images are not incorrectly clustered.
    Details: Prevents the system from incorrectly grouping different scenes together, 
    ensuring a diverse selection of shots is preserved.
    """
    # Create two different images
    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img1_path)
    Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255).save(img2_path)
    
    clusterer = DuplicateClusterer(threshold=0)
    clusters = clusterer.cluster_duplicates([str(img1_path), str(img2_path)])
    
    # Should be 2 different clusters if threshold is low
    assert len(clusters) == 2
