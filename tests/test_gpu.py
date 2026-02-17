import pytest
from src.utils.gpu import detect_gpu_backend

def test_gpu_detection_returns_valid_string():
    """
    Rationale: Ensures the system correctly identifies available hardware (CUDA/MPS/CPU).
    Details: This test is critical for performance optimization, especially for the user's specific 
    hardware (GTX 1650 and future Apple Silicon).
    """
    backend = detect_gpu_backend()
    assert backend in ["cuda", "mps", "cpu"]

def test_gpu_detection_respects_manual_override(mocker):
    """
    Rationale: Validates that users can force a specific hardware backend.
    Details: This test ensures the system honors the `gpu_backend` config setting, allowing for 
    manual CPU/GPU switching during processing or debugging.
    """
    # Mock torch.cuda.is_available to be True
    torch_mock = mocker.patch("torch.cuda.is_available", return_value=True)
    backend = detect_gpu_backend(force_backend=None)
    assert backend == "cuda"
    
    backend_override = detect_gpu_backend(force_backend="cpu")
    assert backend_override == "cpu"
