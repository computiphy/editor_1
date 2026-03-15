"""
Background Removal Engine
=========================
Uses rembg with BiRefNet-Portrait for high-quality human segmentation.
Produces RGBA PNGs with transparent backgrounds.

Supported models (config key: background_removal.model):
  - birefnet-portrait   : Best for people/wedding photos (RECOMMENDED)
  - birefnet-massive    : Highest accuracy, any scene, slower
  - birefnet-general    : Good general-purpose
  - bria-rmbg           : Fast, commercial-grade
  - u2net               : Legacy, fast but low quality for people
  - isnet-general-use   : Alternative general model
"""

import numpy as np
from PIL import Image
from typing import Optional
import os
import platform
import site
import threading
import onnxruntime as ort


# Map config-friendly names to rembg session names
MODEL_MAP = {
    "birefnet-portrait": "birefnet-portrait",
    "birefnet-massive": "birefnet-massive",
    "birefnet-general": "birefnet-general",
    "bria-rmbg": "bria-rmbg",
    "u2net": "u2net",
    "u2netp": "u2netp",
    "isnet-general-use": "isnet-general-use",
}


class BackgroundRemover:
    """Removes backgrounds from images using rembg + BiRefNet."""

    def __init__(self, model: str = "birefnet-portrait", device: str = "cpu"):
        self._lock = threading.Lock()
        self._session = None
        self._model_name = MODEL_MAP.get(model, "birefnet-portrait")
        self._device = device.lower()
        print(f"    BG Removal Model: {self._model_name} (device={self._device})")
        self._dll_handles = []

    def _setup_windows_dll_discovery(self):
        """
        On Windows, manually discover and register NVIDIA DLL directories.
        This bypasses system PATH issues for CUDA, cuDNN, and TensorRT.
        """
        if platform.system() != "Windows":
            return

        # Common search roots for NVIDIA/TensorRT installations
        search_roots = [
            r"C:\Program Files\NVIDIA",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit",
        ]
        
        # We look for 'bin' folders that contain critical DLLs
        critical_dlls = ["nvinfer_10.dll", "cublas64_12.dll", "cudnn64_9.dll"]
        found_dirs = set()

        for root in search_roots:
            if not os.path.exists(root):
                continue
                
            for dirpath, dirnames, filenames in os.walk(root):
                # Optimization: only check 'bin' or 'lib' folders
                folder_name = os.path.basename(dirpath).lower()
                if folder_name not in ["bin", "lib"]:
                    continue
                
                for dll in critical_dlls:
                    if dll in filenames:
                        found_dirs.add(dirpath)
                        break

        # Also add current venv site-packages nvidia bins as a backup
        for s in site.getsitepackages():
            nvidia_path = os.path.join(s, "nvidia")
            if os.path.exists(nvidia_path):
                for dirpath, dirnames, filenames in os.walk(nvidia_path):
                    if os.path.basename(dirpath).lower() == "bin":
                        found_dirs.add(dirpath)

        for d in found_dirs:
            try:
                # 1. os.add_dll_directory is the modern (3.8+) way to handle this
                self._dll_handles.append(os.add_dll_directory(d))
                
                # 2. ALSO update os.environ["PATH"] because some C++ plugins (like ORT's TensorRT)
                # still rely on the old PATH search behavior.
                if d not in os.environ["PATH"]:
                    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

            except Exception:
                pass

    def _get_session(self):
        """Lazy-load the rembg session (downloads model weights on first use)."""
        if self._session is None:
            with self._lock:
                # Double-check inside the lock to prevent a race condition
                if self._session is None:
                    from rembg import new_session
                    
                    if self._device == "tensorrt":
                        self._setup_windows_dll_discovery()
                        from pathlib import Path
                        cache_dir = Path(__file__).resolve().parent.parent.parent / ".trt_engine_cache"
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        
                        providers = [
                            ("TensorrtExecutionProvider", {
                                "trt_engine_cache_enable": True,
                                "trt_engine_cache_path": str(cache_dir)
                            }),
                            ("CUDAExecutionProvider", {
                                "cudnn_conv_algo_search": "HEURISTIC",
                                "arena_extend_strategy": "kNextPowerOfTwo"
                            }),
                            "CPUExecutionProvider"
                        ]
                        os.environ["OMP_NUM_THREADS"] = "1"
                    elif self._device == "gpu":
                        providers = [
                            ("CUDAExecutionProvider", {
                                "cudnn_conv_algo_search": "HEURISTIC",
                                "arena_extend_strategy": "kNextPowerOfTwo"
                            }),
                            "CPUExecutionProvider"
                        ]
                        # VERY IMPORTANT: Restrict ONNX to 1 CPU thread when using GPU.
                        # Otherwise it spawns threads = logical cores, starving Python threads.
                        os.environ["OMP_NUM_THREADS"] = "1"
                        # Restrict to 2 threads for CPU inference to balance load
                        os.environ["OMP_NUM_THREADS"] = "2"
                        
                    self._session = new_session(self._model_name, providers=providers)
        return self._session

    def remove_background(self, image: np.ndarray,
                          post_process_mask: bool = False) -> np.ndarray:
        """
        Remove the background from an RGB image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB, uint8.
            post_process_mask: If True, apply morphological cleanup to the mask
                               to reduce edge artifacts.

        Returns:
            RGBA numpy array (H, W, 4) with transparent background, uint8.
        """
        from rembg import remove

        # Convert numpy → PIL
        pil_img = Image.fromarray(image)

        # Run removal with the selected model session
        result = remove(
            pil_img,
            session=self._get_session(),
            post_process_mask=post_process_mask,
        )

        # Ensure RGBA
        if result.mode != "RGBA":
            result = result.convert("RGBA")

        rgba = np.array(result)

        # Optional: Refine edges with a small Gaussian blur on the alpha channel
        # This softens jagged edges for a more natural cutout
        import cv2
        alpha = rgba[:, :, 3]
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        rgba[:, :, 3] = alpha

        return rgba
