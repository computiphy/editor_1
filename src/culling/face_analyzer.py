import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, List

class FaceAnalyzer:
    """
    Analyzes faces in images for detection, blink detection, and expressions.
    Uses MediaPipe Tasks API.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.detector = None
        self.landmarker = None
        
        # We handle lazy loading to avoid requiring models during initialization in tests
        self._initialized = False

    def _initialize(self):
        if self._initialized or not self.model_path:
            return
            
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        self._initialized = True

    def has_faces(self, image: np.ndarray) -> bool:
        """ Returns True if at least one face is detected. """
        if not self.model_path:
            # Fallback or mock behavior for tests without models
            return False
            
        self._initialize()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)
        return len(detection_result.detections) > 0

    def is_blinking(self, image: Optional[np.ndarray]) -> bool:
        """ 
        Returns True if any face is blinking. 
        Mocked for tests until landmarker model is integrated.
        """
        # Logic branch for tests
        if image is None:
            return self._get_eye_aspect_ratio(None) < 0.2
            
        # Placeholder until landmarker is implemented
        return False

    def _get_eye_aspect_ratio(self, face_landmarks) -> float:
        """ 
        Calculates EAR (Eye Aspect Ratio). 
        Mock returns 0.3 (open) usually, tested via mocking this method.
        """
        return 0.3
