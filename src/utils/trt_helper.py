import os
import json
import threading
from pathlib import Path

class TrtRegistry:
    """Shared utility for managing the human-readable registry of TRT engines."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.registry_path = cache_dir / "registry.json"
        self._lock = threading.Lock()
        
    def get_registry(self):
        """Thread-safe read of the registry."""
        with self._lock:
            if not self.registry_path.exists():
                return {}
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}

    def update(self, model_name: str, engine_filename: str):
        """Thread-safe update of the registry."""
        with self._lock:
            registry = {}
            if self.registry_path.exists():
                try:
                    with open(self.registry_path, "r") as f:
                        registry = json.load(f)
                except Exception:
                    pass
            
            registry[model_name] = engine_filename
            
            # Atomic-ish write: write to temp and rename
            temp_path = self.registry_path.with_suffix(".tmp")
            try:
                with open(temp_path, "w") as f:
                    json.dump(registry, f, indent=4)
                
                # On Windows, replace doesn't work if destination exists
                if self.registry_path.exists():
                    os.remove(self.registry_path)
                os.rename(temp_path, self.registry_path)
            except Exception as e:
                if temp_path.exists():
                    os.remove(temp_path)
                raise e

    def get_engine_for_model(self, model_name: str):
        """Returns the engine filename for a model if it exists in the registry."""
        registry = self.get_registry()
        filename = registry.get(model_name)
        if filename and (self.cache_dir / filename).exists():
            return filename
        return None

    def get_snapshot(self):
        """Returns a set of engine filenames currently in the cache."""
        return {f.name for f in self.cache_dir.glob("*.engine")}
