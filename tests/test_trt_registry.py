import os
import shutil
import json
import pytest
import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
sys.path.append(os.getcwd())

from src.segmentation.background_remover import BackgroundRemover

def test_registry_sync_and_accurate_mapping():
    """
    Test that BackgroundRemover updates the registry accurately.
    Uses u2net as it's small/fast.
    """
    cache_dir = Path(".trt_engine_cache")
    registry_path = cache_dir / "registry.json"
    model_name = "u2net"
    
    # 1. SETUP: Clean state
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
        
    # Create a backup directory
    backup_dir = Path(".trt_engine_cache_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir()
    
    # Copy existing registry and engines to backup
    if registry_path.exists():
        shutil.copy(str(registry_path), str(backup_dir))
        os.remove(registry_path)
    
    # Copy all engines and remove from active dir for a clean test
    original_engines = list(cache_dir.glob("*.engine"))
    for engine in original_engines:
        shutil.copy(str(engine), str(backup_dir))
        os.remove(engine)

    try:
        # 2. RUN: Main pipeline simulation (BackgroundRemover)
        remover = BackgroundRemover(model=model_name, device="tensorrt")
        remover._get_session()
        
        # 3. VERIFY
        assert registry_path.exists(), "BackgroundRemover did not create/update registry.json"
        
        with open(registry_path, "r") as f:
            registry = json.load(f)
            
        assert model_name in registry, f"'{model_name}' not found in registry"
        
        engine_filename = registry[model_name]
        engine_path = cache_dir / engine_filename
        assert engine_path.exists(), f"Cached engine file {engine_filename} missing from disk"
        
    finally:
        # CLEANUP: Restore everything from backup and remove test engine
        # Remove any engine created by the test to avoid junk
        if registry_path.exists():
            with open(registry_path, "r") as f:
                reg = json.load(f)
                if model_name in reg:
                    test_engine = cache_dir / reg[model_name]
                    if test_engine.exists():
                        os.remove(test_engine)
            os.remove(registry_path)

        # Restore original files
        for f in backup_dir.glob("*"):
            shutil.move(str(f), str(cache_dir))
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

if __name__ == "__main__":
    pytest.main([__file__])
