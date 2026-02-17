import numpy as np
from typing import Optional, Dict, Any
from src.core.enums import RestorationMode
from src.utils.gpu import detect_gpu_backend

class RestorationEngine:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or detect_gpu_backend()
        self.nafnet = None
        self.gfpgan = None
        self.swinir = None

    def _init_nafnet(self):
        if self.nafnet is None:
            from src.restoration.nafnet_restore import NAFNetRestorer
            self.nafnet = NAFNetRestorer(backend=self.backend)

    def _init_gfpgan(self):
        if self.gfpgan is None:
            from src.restoration.gfpgan_restore import GFPGANRestorer
            self.gfpgan = GFPGANRestorer(backend=self.backend)

    def restore(
        self, 
        image: np.ndarray, 
        has_faces: bool = False, 
        blur_score: float = 100.0,
        mode: RestorationMode = RestorationMode.AUTO
    ) -> np.ndarray:
        """
        Routes the image to the appropriate restoration model(s).
        """
        restored = image.copy()
        
        # Simple routing logic
        if has_faces:
            self._init_gfpgan()
            restored = self.gfpgan.restore(restored)
            
        if blur_score < 100.0: # Threshold for "is blurry"
            self._init_nafnet()
            restored = self.nafnet.restore(restored)
            
        return restored
