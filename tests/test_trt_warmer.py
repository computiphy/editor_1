import os
import subprocess
import sys
import pytest
from pathlib import Path

def test_trt_warmer_cli_and_output():
    """
    Test that the warmer script can be called and produces a cache file.
    Note: We use a small model (u2net) for the test to avoid massive downloads.
    """
    script_path = Path("scripts/warm_trt_cache.py")
    cache_dir = Path(".trt_engine_cache")
    
    # Ensure script exists (will fail until Green phase)
    assert script_path.exists(), "Warmer script does not exist"
    
    # Run the script for u2net
    # We expect this to fail or do nothing until implemented
    result = subprocess.run(
        [sys.executable, str(script_path), "u2net"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
    
    # Verify that at least one .engine file was created/updated in the cache
    # (Checking for specific matching might be hard due to hashes, but we check directory state)
    engines = list(cache_dir.glob("*.engine"))
    assert len(engines) > 0, "No TensorRT engines were found in cache after running warmer"

if __name__ == "__main__":
    pytest.main([__file__])
