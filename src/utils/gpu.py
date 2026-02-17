import torch
from typing import Optional

def detect_gpu_backend(force_backend: Optional[str] = None) -> str:
    """ Detects the available GPU backend or returns the forced one. """
    if force_backend in ["cuda", "mps", "cpu"]:
        return force_backend
        
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device(backend: Optional[str] = None) -> torch.device:
    """ Returns a torch.device object for the selected backend. """
    selected = detect_gpu_backend(backend)
    if selected == "cuda":
        return torch.device("cuda")
    elif selected == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")
