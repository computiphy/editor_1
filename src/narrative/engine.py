import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any
from src.core.models import Chapter

class NarrativeEngine:
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples

    def group_to_chapters(self, image_paths: List[str]) -> List[Chapter]:
        """
        Groups images into chapters based on visual similarity (CLIP embeddings).
        """
        if not image_paths:
            return []

        embeddings = self._compute_embeddings(image_paths)
        
        # Cluster using DBSCAN
        # Normalizing embeddings first is usually better for cosine similarity
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Metric 'cosine' is good for CLIP
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine').fit(norm_embeddings)
        labels = clustering.labels_

        chapters = []
        unique_labels = set(labels)
        
        for label in sorted(unique_labels):
            if label == -1:
                # Noise/Outliers: each gets its own "Uncategorized" chapter or we group them
                noise_indices = np.where(labels == -1)[0]
                for idx in noise_indices:
                    chapters.append(Chapter(
                        name=f"Snippet {len(chapters)+1}",
                        images=[image_paths[idx]],
                        confidence=0.5
                    ))
                continue

            indices = np.where(labels == label)[0]
            chapters.append(Chapter(
                name=f"Chapter {len(chapters)+1}",
                images=[image_paths[idx] for idx in indices],
                confidence=1.0
            ))

        return chapters

    def _compute_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Compute CLIP embeddings. Placeholder for tests.
        """
        # In real implementation:
        # import open_clip
        # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        return np.random.rand(len(image_paths), 512)
