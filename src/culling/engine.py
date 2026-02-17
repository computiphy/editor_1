from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from src.core.models import ImageScore, Cluster
from src.culling.blur_detector import BlurDetector
from src.culling.face_analyzer import FaceAnalyzer
from src.culling.quality_assessor import QualityAssessor
from src.culling.duplicate_cluster import DuplicateClusterer
from src.utils.image_io import load_image

class CullingEngine:
    """
    Orchestrates the multi-stage culling process:
    1. Technical filtering (Blur)
    2. Model-based assessment (BRISQUE/NIQE)
    3. Face intelligence (Landmarks, EAR, Expression)
    4. Near-duplicate clustering
    """
    def __init__(
        self,
        blur_detector: BlurDetector,
        face_analyzer: FaceAnalyzer,
        quality_assessor: QualityAssessor,
        clusterer: DuplicateClusterer,
        config: Dict[str, Any] = None
    ):
        self.blur_detector = blur_detector
        self.face_analyzer = face_analyzer
        self.quality_assessor = quality_assessor
        self.clusterer = clusterer
        self.config = config or {}

    def evaluate_image(self, path: Path) -> ImageScore:
        """ Evaluate a single image and return a score. """
        image = load_image(str(path))
        
        blur_score = self.blur_detector.calculate_variance(image)
        is_blurry = self.blur_detector.is_blurry(image)
        
        brisque, niqe = self.quality_assessor.calculate_scores(image)
        
        has_faces = self.face_analyzer.has_faces(image)
        is_blinking = False
        if has_faces:
            is_blinking = self.face_analyzer.is_blinking(image)
            
        # Composite logic (example thresholds)
        passed = True
        reasons = []
        
        if is_blurry:
            passed = False
            reasons.append("blurry")
        if is_blinking:
            passed = False
            reasons.append("blink_detected")
            
        # Quality thresholding (BRISQUE lower is better)
        if brisque > 70.0:
            passed = False
            reasons.append("perceptual_quality_low")

        return ImageScore(
            path=path,
            blur_score=float(blur_score),
            fft_energy=0.0, # Placeholder
            brisque_score=float(brisque),
            niqe_score=float(niqe),
            has_faces=has_faces,
            blink_detected=is_blinking,
            expression_score=0.8, # Placeholder
            overall_quality=0.9 if passed else 0.4,
            passed=passed,
            rejection_reasons=reasons
        )

    def cluster_and_filter(self, paths: List[Path]) -> List[Cluster]:
        """ Cluster images and identify primary shots. """
        return self.clusterer.cluster_duplicates([str(p) for p in paths])
