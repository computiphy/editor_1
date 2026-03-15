import pytest
import os
import onnxruntime as ort
from src.segmentation.background_remover import BackgroundRemover

def test_tensorrt_execution_provider_loads():
    """
    Strict TDD Test: Verify that TensorrtExecutionProvider can be initialized.
    This test will fail (🔴 Red) if nvinfer_10.dll or its dependencies are missing
    from the perspective of the onnxruntime C++ bridge.
    """
    remover = BackgroundRemover(model="u2net", device="tensorrt")
    
    # Attempting to load the session should NOT fail with Error 126
    try:
        session = remover._get_session()
        
        # Verify that TensorrtExecutionProvider is actually in the session's providers
        providers = session.get_providers()
        assert "TensorrtExecutionProvider" in providers, f"Expected TensorrtExecutionProvider but got {providers}"
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        available = ort.get_available_providers()
        print(f"\nCaught Exception: {error_msg}")
        print(f"Available Providers in OS: {available}")
        traceback.print_exc()
        
        if "nvinfer_10.dll" in error_msg or "Error 126" in error_msg:
            pytest.fail(f"🔴 RED: TensorRT DLL loading failed: {error_msg}")
        elif "TensorrtExecutionProvider" in error_msg and "is not in available provider names" in error_msg:
             pytest.fail(f"🔴 RED: TensorRT provider not available: {error_msg}. Available: {available}")
        else:
            pytest.fail(f"Failure during TensorRT init: {error_msg}")

if __name__ == "__main__":
    # Allow running directly for quick feedback
    try:
        test_tensorrt_execution_provider_loads()
        print("🟢 TEST PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
