import os
import sys
import argparse
import platform
import site
import numpy as np
import json
import psutil
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# NOTE: This script is deliberately standalone and does not import from src/.
# This ensures it can be run for environment setup without affecting the main project development.

class StandaloneTrtRegistry:
    """Standalone version of the registry logic to keep this script independent."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.registry_path = cache_dir / "registry.json"
        
    def get_registry(self):
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def update(self, model_name: str, engine_filename: str):
        registry = self.get_registry()
        registry[model_name] = engine_filename
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=4)

    def get_snapshot(self):
        return {f.name for f in self.cache_dir.glob("*.engine")}

def setup_windows_dll_discovery():
    """Manually discover and register NVIDIA DLL directories on Windows."""
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

    # Add venv site-packages nvidia bins
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

def warm_model(model_name: str, cache_dir: Path, threads: int, registry: StandaloneTrtRegistry):
    """Triggers TensorRT engine compilation for a specific model via rembg."""
    try:
        import onnxruntime as ort
        from rembg import new_session, remove
        
        setup_windows_dll_discovery()
        
        # Snapshot state BEFORE warming
        old_files = registry.get_snapshot()
        
        # Explicit Multi-Threading Configuration
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = threads
        sess_opts.inter_op_num_threads = threads
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir)
            }),
            ("CUDAExecutionProvider", {
                "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kNextPowerOfTwo"
            }),
            "CPUExecutionProvider"
        ]
        
        # Instantiate session with TRT
        session = new_session(model_name, providers=providers)
        
        # Verify provider
        if hasattr(session, 'inner_session'):
            ort_session = session.inner_session
        else:
            ort_session = session
            
        providers = ort_session.get_providers()
        if "TensorrtExecutionProvider" in providers:
            # Run one dummy inference to ensure kernels are built
            dummy_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            remove(dummy_img, session=session)
            
            # Snapshot state AFTER warming to find the EXACT engine file
            new_files = registry.get_snapshot()
            diff = new_files - old_files
            
            if diff:
                # We found new engines!
                engine_file = list(diff)[0] # Usually just one
                registry.update(model_name, engine_file)
                return True, engine_file
            else:
                # No NEW file, maybe it was an update or already existed
                # Fallback to newest if we're sure it worked
                engines = list(cache_dir.glob("*.engine"))
                if engines:
                    newest = max(engines, key=os.path.getmtime)
                    registry.update(model_name, newest.name)
                    return True, newest.name
            
            return True, None
        else:
            return False, f"TensorrtExecutionProvider missing. Available: {providers}"
            
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Standalone TensorRT Cache Warmer.")
    parser.add_argument("models", nargs="+", help="Model names (e.g., birefnet-portrait, u2net)")
    args = parser.parse_args()

    # Optimal thread count for compilation: Use all logical cores
    cpu_cores = os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    
    cache_dir = Path(".trt_engine_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    registry = StandaloneTrtRegistry(cache_dir)
    reg_data = registry.get_registry()

    print(f"--- Standalone TensorRT Cache Warmer ---")
    print(f"Optimal Threads: {cpu_cores}")
    
    if reg_data:
        print(f"Detected cached models: {len(reg_data)}")
        for model, engine in reg_data.items():
            print(f"  - {model} -> {engine}")
    else:
        print("No localized model registry found in .trt_engine_cache/")

    print(f"Models to warm: {', '.join(args.models)}")
    print("-" * 30)

    peak_usage = 0.0
    
    for model_name in tqdm(args.models, desc="Warming models"):
        if model_name in reg_data:
            engine_path = cache_dir / reg_data[model_name]
            if engine_path.exists():
                print(f"  [SKIP] '{model_name}' is already cached as {reg_data[model_name]}")
                continue
            
        success, res = warm_model(model_name, cache_dir, cpu_cores, registry)
        
        peak_usage = max(peak_usage, max(psutil.cpu_percent(interval=None, percpu=True)))

        if success:
            print(f"  [SUCCESS] Produced engine for '{model_name}': {res}")
        else:
            print(f"  [ERROR] Failed to warm '{model_name}': {res}")

    print("-" * 30)
    print(f"Warming process complete.")
    print(f"Peak per-core utilization detected: {peak_usage}%")

if __name__ == "__main__":
    main()
