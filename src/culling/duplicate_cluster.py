import imagehash
from PIL import Image
from pathlib import Path
from typing import List
from src.core.models import Cluster

class DuplicateClusterer:
    def __init__(self, threshold: int = 5):
        self.threshold = threshold

    def cluster_duplicates(self, image_paths: List[str]) -> List[Cluster]:
        """ Cluster near-duplicate images using pHash and Hamming distance. """
        if not image_paths:
            return []

        # Calculate hashes
        hashes = {}
        for path in image_paths:
            img = Image.open(path)
            hashes[path] = imagehash.phash(img)

        # Simple adjacency-based clustering (connected components)
        visited = set()
        clusters = []

        path_list = list(hashes.keys())
        for i, path in enumerate(path_list):
            if path in visited:
                continue

            current_cluster_paths = [Path(path)]
            visited.add(path)
            
            # Find all neighbors (within threshold)
            # This is a simple greedy approach for near-duplicates
            for j in range(i + 1, len(path_list)):
                other_path = path_list[j]
                if other_path in visited:
                    continue
                
                if hashes[path] - hashes[other_path] <= self.threshold:
                    current_cluster_paths.append(Path(other_path))
                    visited.add(other_path)

            clusters.append(Cluster(
                cluster_id=f"cluster_{len(clusters)}",
                images=current_cluster_paths,
                representative=current_cluster_paths[0], # Simplification
                hash_distances={}
            ))

        return clusters
