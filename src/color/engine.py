import cv2
import numpy as np
from src.core.enums import ColorMethod

class ColorGradingEngine:
    def __init__(self, method: ColorMethod = ColorMethod.LAB_STATISTICAL):
        self.method = method

    def apply_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Applies color transfer from target to source.
        Reinhard et al. (2001) method in Lab color space.
        """
        if self.method == ColorMethod.LAB_STATISTICAL:
            return self._reinhard_transfer(source, target)
        return source

    def _reinhard_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Convert to Lab
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2Lab).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2Lab).astype(np.float32)

        # Compute mean and std for both
        s_mean, s_std = self._get_stats(source_lab)
        t_mean, t_std = self._get_stats(target_lab)

        # Transfer
        res_lab = (source_lab - s_mean) * (t_std / s_std) + t_mean
        res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)

        # Convert back
        return cv2.cvtColor(res_lab, cv2.COLOR_Lab2RGB)

    def _get_stats(self, img_lab: np.ndarray):
        (l, a, b) = cv2.split(img_lab)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
        
        # Avoid division by zero
        lStd = max(lStd, 1e-5)
        aStd = max(aStd, 1e-5)
        bStd = max(bStd, 1e-5)
        
        return np.array([lMean, aMean, bMean]), np.array([lStd, aStd, bStd])
