import os
import sys
import argparse
import platform
import site
import numpy as np
import json
import psutil
import time
import threading
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional, Any

# --- Interfaces (DIP/ISP) ---

class IModelRegistry(ABC):
    @abstractmethod
    def get_registry(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def update(self, model_name: str, engine_filename: str):
        pass

    @abstractmethod
    def get_snapshot(self) -> Set[str]:
        pass

class IEngineWarmer(ABC):
    @abstractmethod
    def warm(self, model_name: str, threads: int) -> Tuple[bool, Any]:
        pass

# --- Concrete Implementations (SRP) ---

class StandaloneTrtRegistry(IModelRegistry):
    """File-based registry with atomic-ish updates and basic locking."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.registry_path = cache_dir / "registry.json"
        self._lock_path = cache_dir / "registry.lock"
        
    def _acquire_lock(self, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            try:
                # Use exclusive file creation as a lock
                with open(self._lock_path, "x"):
                    return True
            except FileExistsError:
                time.sleep(0.1)
        return False

    def _release_lock(self):
        if self._lock_path.exists():
            try:
                os.remove(self._lock_path)
            except Exception:
                pass

    def get_registry(self) -> Dict[str, str]:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def update(self, model_name: str, engine_filename: str):
        if not self._acquire_lock():
            # If we can't get the lock, try anyway but log it (non-ideal but keeps process moving)
            print(f"  [WARN] Could not acquire registry lock for {model_name}")
            
        try:
            registry = self.get_registry()
            registry[model_name] = engine_filename
            
            # Atomic rename pattern
            temp_path = self.registry_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(registry, f, indent=4)
            
            if self.registry_path.exists():
                os.remove(self.registry_path)
            os.rename(temp_path, self.registry_path)
        finally:
            self._release_lock()

    def get_snapshot(self) -> Set[str]:
        return {f.name for f in self.cache_dir.glob("*.engine")}

class RembgEngineWarmer(IEngineWarmer):
    """Warms engines using the rembg library."""
    def __init__(self, cache_dir: Path, registry: IModelRegistry):
        self.cache_dir = cache_dir
        self.registry = registry

    def warm(self, model_name: str, threads: int) -> Tuple[bool, Any]:
        try:
            import onnxruntime as ort
            from rembg import new_session, remove
            
            DllDiscovery.setup()
            
            old_files = self.registry.get_snapshot()
            
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = threads
            sess_opts.inter_op_num_threads = threads
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(self.cache_dir)
                }),
                ("CUDAExecutionProvider", {
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "arena_extend_strategy": "kNextPowerOfTwo"
                }),
                "CPUExecutionProvider"
            ]
            
            session = new_session(model_name, providers=providers)
            
            if hasattr(session, 'inner_session'):
                ort_session = session.inner_session
            else:
                ort_session = session
                
            active_providers = ort_session.get_providers()
            if "TensorrtExecutionProvider" in active_providers:
                dummy_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
                remove(dummy_img, session=session)
                
                new_files = self.registry.get_snapshot()
                diff = new_files - old_files
                
                if diff:
                    engine_file = list(diff)[0]
                    self.registry.update(model_name, engine_file)
                    return True, engine_file
                else:
                    engines = list(self.cache_dir.glob("*.engine"))
                    if engines:
                        newest = max(engines, key=os.path.getmtime)
                        self.registry.update(model_name, newest.name)
                        return True, newest.name
                
                return True, None
            else:
                return False, f"TensorrtExecutionProvider missing. Available: {active_providers}"
                
        except Exception as e:
            return False, str(e)

class DllDiscovery:
    """Handles Windows DLL discovery logic."""
    @staticmethod
    def setup():
        if platform.system() != "Windows":
            return

        search_roots = [
            r"C:\Program Files\NVIDIA",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit",
        ]
        
        critical_dlls = ["nvinfer_10.dll", "cublas64_12.dll", "cudnn64_9.dll"]
        found_dirs = set()

        for root in search_roots:
            if not os.path.exists(root):
                continue
                
            for dirpath, dirnames, filenames in os.walk(root):
                folder_name = os.path.basename(dirpath).lower()
                if folder_name not in ["bin", "lib"]:
                    continue
                
                for dll in critical_dlls:
                    if dll in filenames:
                        found_dirs.add(dirpath)
                        break

        for s in site.getsitepackages():
            nvidia_path = os.path.join(s, "nvidia")
            if os.path.exists(nvidia_path):
                for dirpath, dirnames, filenames in os.walk(nvidia_path):
                    if os.path.basename(dirpath).lower() == "bin":
                        found_dirs.add(dirpath)

        for d in found_dirs:
            try:
                os.add_dll_directory(d)
                if d not in os.environ["PATH"]:
                    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            except Exception:
                pass

# --- Parallel Orchestrator (SRP/OCP) ---

class ParallelWarmer:
    """Orchestrates parallel warming of multiple models."""
    def __init__(self, models: List[str], base_warmer: IEngineWarmer, registry: IModelRegistry, max_workers: int):
        self.models = models
        self.base_warmer = base_warmer
        self.registry = registry
        self.max_workers = max_workers
        self.cpu_cores = os.cpu_count() or 1
        # Distribute threads: fewer threads per model if running multiple models
        self.threads_per_worker = max(1, self.cpu_cores // max_workers)

    def run(self) -> Dict[str, Tuple[bool, Any]]:
        results = {}
        
        # Filter out already cached models to avoid redundant work in subprocesses
        reg_data = self.registry.get_registry()
        models_to_build = []
        for m in self.models:
            if m in reg_data and (self.registry.cache_dir / reg_data[m]).exists():
                print(f"  [SKIP] '{m}' is already cached as {reg_data[m]}")
                results[m] = (True, reg_data[m])
            else:
                models_to_build.append(m)

        if not models_to_build:
            return results

        if self.max_workers == 1 or len(models_to_build) == 1:
            print(f"Building {len(models_to_build)} models sequentially...")
            for m in tqdm(models_to_build, desc="Sequential Build"):
                success, res = self.base_warmer.warm(m, self.threads_per_worker)
                results[m] = (success, res)
                if success:
                    print(f"  [SUCCESS] '{m}': {res}")
                else:
                    print(f"  [ERROR] '{m}': {res}")
            return results

        print(f"Building {len(models_to_build)} models in parallel using {self.max_workers} workers...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:

            # We must use a top-level function or a class method that can be pickled for the sub-process
            future_to_model = {
                executor.submit(_warm_worker, m, self.threads_per_worker, type(self.base_warmer), str(self.registry.cache_dir)): m 
                for m in models_to_build
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_model), total=len(future_to_model), desc="Parallel Build"):
                model = future_to_model[future]
                try:
                    success, res = future.result()
                    results[model] = (success, res)
                    if success:
                        print(f"  [SUCCESS] '{model}': {res}")
                    else:
                        print(f"  [ERROR] '{model}': {res}")
                except Exception as exc:
                    results[model] = (False, str(exc))
                    print(f"  [CRITICAL] '{model}' generated an exception: {exc}")
                    
        return results

def _warm_worker(model_name: str, threads: int, warmer_class: type, cache_dir_str: str) -> Tuple[bool, Any]:
    """Top-level helper for ProcessPoolExecutor to avoid pickling issues."""
    cache_dir = Path(cache_dir_str)
    # We re-instantiate within the process to be safe
    registry = StandaloneTrtRegistry(cache_dir)
    warmer = warmer_class(cache_dir, registry)
    return warmer.warm(model_name, threads)

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Parallel TensorRT Cache Warmer.")
    parser.add_argument("models", nargs="+", help="Model names (e.g., birefnet-portrait, u2net)")
    parser.add_argument("--parallel", type=int, default=2, help="Number of models to build concurrently (default: 2)")
    args = parser.parse_args()

    # Optimal thread count for compilation: Use all logical cores for OMP if possible
    cpu_cores = os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    
    cache_dir = Path(".trt_engine_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    registry = StandaloneTrtRegistry(cache_dir)
    reg_data = registry.get_registry()

    print(f"--- Parallel TensorRT Cache Warmer ---")
    print(f"System Cores: {cpu_cores}, Parallel Workers: {args.parallel}")
    
    if reg_data:
        print(f"Detected cached models: {len(reg_data)}")
        for model, engine in reg_data.items():
            print(f"  - {model} -> {engine}")
    else:
        print("No localized model registry found in .trt_engine_cache/")

    print(f"Models to process: {', '.join(args.models)}")
    print("-" * 30)

    start_time = time.time()
    
    warmer = RembgEngineWarmer(cache_dir, registry)
    orchestrator = ParallelWarmer(args.models, warmer, registry, args.parallel)
    results = orchestrator.run()

    end_time = time.time()
    print("-" * 30)
    print(f"Warming process complete in {end_time - start_time:.2f}s.")

if __name__ == "__main__":
    main()
