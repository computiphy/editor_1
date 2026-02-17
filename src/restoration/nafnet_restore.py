import torch
from src.utils.tiling import ImageTiler
from src.utils.gpu import get_device

class NAFNetRestorer:
    def __init__(self, backend: str = "auto", model_path: str = "weights/NAFNet-DeBlur.pth"):
        self.device = get_device(backend)
        self.model_path = model_path
        self._model = None
        self.tiler = ImageTiler(tile_size=512, overlap=64)

    def _load_model(self):
        if self._model is None:
            # Note: In a real implementation, we would import the NAFNet architecture class
            # For this MVP, we use a placeholder that simulates the forward pass logic
            # to keep the repository focused on the pipeline architecture.
            self._model = torch.nn.Identity() # Placeholder
            self._model.to(self.device).eval()

    def restore(self, image: np.ndarray) -> np.ndarray:
        """
        NAFNet restoration with tiling for Low VRAM (GTX 1650).
        """
        self._load_model()
        
        # Split into tiles to stay within 4GB VRAM
        tiles, metadata = self.tiler.split(image)
        restored_tiles = []
        
        with torch.no_grad():
            for tile in tiles:
                # To Tensor (CHW, Normalized)
                t = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                t = t.unsqueeze(0).to(self.device)
                
                # Inference
                out = self._model(t)
                
                # Back to NumPy
                out = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
                restored_tiles.append((out * 255.0).clip(0, 255).astype(np.uint8))
        
        # Stitch back together
        return self.tiler.merge(restored_tiles, metadata)
