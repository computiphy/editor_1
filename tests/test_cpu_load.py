import os
import subprocess
import sys
import time
import psutil
import pytest
from pathlib import Path
import json

def monitor_cpu_load(duration, interval=0.1):
    """Monitors per-CPU load for a duration and returns the peak per-core utilization."""
    peaks = [0.0] * psutil.cpu_count()
    start_time = time.time()
    while time.time() - start_time < duration:
        current_load = psutil.cpu_percent(interval=None, percpu=True)
        for i, load in enumerate(current_load):
            if load > peaks[i]:
                peaks[i] = load
        time.sleep(interval)
    return peaks

def test_warmer_utilizes_multiple_cores():
    """
    Test that the warmer script hits multiple logical cores during compilation.
    And verifies that a registry.json is created.
    """
    script_path = Path("scripts/warm_trt_cache.py")
    cache_dir = Path(".trt_engine_cache")
    registry_path = cache_dir / "registry.json"
    
    # Ensure a clean state for the specific model we test
    # (Using u2netp for a faster test if possible, or just u2net)
    model_to_test = "u2net"
    
    # Remove existing engine for this model if it exists to force compilation
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
        if model_to_test in registry:
            engine_file = cache_dir / registry[model_to_test]
            if engine_file.exists():
                os.remove(engine_file)
            del registry[model_to_test]
            with open(registry_path, "w") as f:
                json.dump(registry, f)

    # Start the warmer in a subprocess
    # We'll monitor for 10 seconds while it builds (compilation is usually the slowest part)
    process = subprocess.Popen([sys.executable, str(script_path), model_to_test])
    
    # Monitor CPU peaks
    peaks = monitor_cpu_load(duration=15)
    
    process.wait()
    assert process.returncode == 0
    
    # Check for multi-core usage: At least 4 cores (if available) should spike > 30%
    # or more than 50% on at least 2 cores.
    high_load_cores = [p for p in peaks if p > 30.0]
    num_cores = psutil.cpu_count()
    
    print(f"\nPeak per-core loads: {peaks}")
    assert len(high_load_cores) >= min(4, num_cores), f"Only {len(high_load_cores)} cores showed significant load. Expected more for multi-threaded compilation."

def test_warmer_produces_registry():
    """Verifies that the warmer utility produces a registry.json file."""
    registry_path = Path(".trt_engine_cache/registry.json")
    # This should fail if registry.json is not yet implemented
    assert registry_path.exists(), "registry.json was not created by the warmer"
    
    with open(registry_path, "r") as f:
        registry = json.load(f)
    assert len(registry) > 0, "registry.json is empty"

if __name__ == "__main__":
    pytest.main([__file__])
