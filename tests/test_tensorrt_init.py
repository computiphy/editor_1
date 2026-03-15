import pytest
import os
import onnxruntime as ort
from src.segmentation.background_remover import BackgroundRemover

def test_tensorrt_execution_provider_loads():
    """
    Strict TDD Test: Verify that TensorrtExecutionProvider can be initialized.
    This test will fail (Red) if nvinfer_10.dll or its dependencies are missing
    from the perspective of the onnxruntime C++ bridge.
    """
    remover = BackgroundRemover(model="u2net", device="tensorrt")
    session = remover._get_session()
    
    # Verify that TensorrtExecutionProvider is actually in the session's providers
    # rembg sessions often wrap the ORT session in .inner_session or just are the session
    if hasattr(session, 'inner_session'):
        ort_session = session.inner_session
    else:
        ort_session = session
        
    providers = ort_session.get_providers()
    assert "TensorrtExecutionProvider" in providers, f"Expected TensorrtExecutionProvider but got {providers}"

if __name__ == "__main__":
    # Allow running directly for quick feedback
    try:
        test_tensorrt_execution_provider_loads()
        print("TEST PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
