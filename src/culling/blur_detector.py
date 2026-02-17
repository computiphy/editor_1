import cv2
import numpy as np

class BlurDetector:
    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold

    def calculate_variance(self, image: np.ndarray) -> float:
        """ Calculate the variance of the Laplacian. """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def is_blurry(self, image: np.ndarray) -> bool:
        return self.calculate_variance(image) < self.threshold
