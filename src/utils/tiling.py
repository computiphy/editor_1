import numpy as np
from typing import List, Tuple, Dict, Any

class ImageTiler:
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap

    def split(self, image: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Splits an image into overlapping tiles.
        """
        h, w, c = image.shape
        stride = self.tile_size - self.overlap
        
        tiles = []
        positions = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Clamp to img bounds
                y1 = min(y, h - self.tile_size) if h > self.tile_size else 0
                x1 = min(x, w - self.tile_size) if w > self.tile_size else 0
                y2 = min(y1 + self.tile_size, h)
                x2 = min(x1 + self.tile_size, w)
                
                tile = image[y1:y2, x1:x2]
                tiles.append(tile)
                positions.append((y1, x1, y2, x2))
                
                if x1 + self.tile_size >= w: break
            if y1 + self.tile_size >= h: break
            
        metadata = {
            "shape": image.shape,
            "positions": positions
        }
        return tiles, metadata

    def merge(self, tiles: List[np.ndarray], metadata: Dict[str, Any]) -> np.ndarray:
        """
        Merges overlapping tiles back into a single image.
        Simple implementation: Overwrites with the latest tile (no blending for now).
        """
        h, w, c = metadata["shape"]
        positions = metadata["positions"]
        
        reconstructed = np.zeros((h, w, c), dtype=tiles[0].dtype)
        
        for tile, (y1, x1, y2, x2) in zip(tiles, positions):
            reconstructed[y1:y2, x1:x2] = tile
            
        return reconstructed
