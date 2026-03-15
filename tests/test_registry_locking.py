import os
import time
import concurrent.futures
from pathlib import Path
from scripts.warm_trt_cache import StandaloneTrtRegistry

def test_registry_locking_prevents_corruption():
    """
    Verifies that StandaloneTrtRegistry handles concurrent updates from multiple processes
    using its file-based locking mechanism.
    """
    cache_dir = Path(".trt_engine_cache_test_lock")
    cache_dir.mkdir(parents=True, exist_ok=True)
    registry_path = cache_dir / "registry.json"
    
    # Ensure clean state
    if registry_path.exists():
        os.remove(registry_path)
        
    num_updates = 10
    models = [f"model_{i}" for i in range(num_updates)]
    
    def update_worker(model_name):
        reg = StandaloneTrtRegistry(cache_dir)
        reg.update(model_name, f"engine_{model_name}.engine")
        return True

    # Use ThreadPoolExecutor to simulate concurrent calls (Registry uses file lock which is process-safe too)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(update_worker, models))
        
    # Verify all models are present in the registry
    reg = StandaloneTrtRegistry(cache_dir)
    data = reg.get_registry()
    
    assert len(data) == num_updates
    for m in models:
        assert m in data
        assert data[m] == f"engine_{m}.engine"
        
    # Cleanup
    if registry_path.exists():
        os.remove(registry_path)
    if (cache_dir / "registry.lock").exists():
        os.remove(cache_dir / "registry.lock")
    cache_dir.rmdir()

if __name__ == "__main__":
    test_registry_locking_prevents_corruption()
    print("SUCCESS")
