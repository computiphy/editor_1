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

def get_engine_from_cache(model_name: str, cache_dir: Path):
    """Finds the most recent engine file for a given model in the cache."""
    # Since engine files are hash-based, we look for matches in registry.json first
    registry_path = cache_dir / "registry.json"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
            if model_name in registry:
                engine_file = cache_dir / registry[model_name]
                if engine_file.exists():
                    return registry[model_name]
    
    # Fallback: crude scan (not recommended but for robustness)
    engines = [f.name for f in cache_dir.glob("*.engine") if f.name.startswith("TensorrtExecutionProvider_TRTKernel")]
    # Note: TRT engine names usually contain hashes, not model names directly.
    # Without the registry, we can't be sure which hash belongs to which model perfectly.
    return None

def update_registry(model_name: str, cache_dir: Path):
    """Updates registry.json by detecting the newest engine file."""
    registry_path = cache_dir / "registry.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
            
    # Find the newest .engine file in the cache
    engines = list(cache_dir.glob("*.engine"))
    if not engines:
        return
        
    newest_engine = max(engines, key=os.path.getmtime)
    registry[model_name] = newest_engine.name
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

def warm_model(model_name: str, cache_dir: Path, threads: int):
    """Triggers TensorRT engine compilation for a specific model via rembg."""
    try:
        import onnxruntime as ort
        from rembg import new_session, remove
        
        setup_windows_dll_discovery()
        
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
            # Record in registry
            update_registry(model_name, cache_dir)
            return True, None
        else:
            return False, f"TensorrtExecutionProvider missing. Available: {providers}"
            
    except Exception as e:
        return False, str(e)

def monitor_load(stop_event):
    """Background load monitor."""
    peak = 0.0
    while not stop_event[0]:
        load = psutil.cpu_percent(interval=0.5)
        if load > peak:
            peak = load
    return peak

def main():
    parser = argparse.ArgumentParser(description="Standalone TensorRT Cache Warmer.")
    parser.add_argument("models", nargs="+", help="Model names (e.g., birefnet-portrait, u2net)")
    args = parser.parse_args()

    # Optimal thread count for compilation: Use all logical cores
    cpu_cores = os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    
    cache_dir = Path(".trt_engine_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = cache_dir / "registry.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)

    print(f"--- Standalone TensorRT Cache Warmer ---")
    print(f"Optimal Threads: {cpu_cores}")
    
    if registry:
        print(f"Detected cached models: {len(registry)}")
        for model, engine in registry.items():
            print(f"  - {model} -> {engine}")
    else:
        print("No localized model registry found in .trt_engine_cache/")

    print(f"Models to warm: {', '.join(args.models)}")
    print("-" * 30)

    # Start CPU monitoring
    peak_usage = 0.0
    
    # We'll use a simple background monitoring thread (simulated here)
    def get_peak():
        return max(psutil.cpu_percent(interval=0.1, percpu=True))

    for model_name in tqdm(args.models, desc="Warming models"):
        if model_name in registry:
            engine_path = cache_dir / registry[model_name]
            if engine_path.exists():
                print(f"  [SKIP] '{model_name}' is already cached as {registry[model_name]}")
                continue
            
        # Start load tracking for this model
        success, err = warm_model(model_name, cache_dir, cpu_cores)
        
        peak_usage = max(peak_usage, max(psutil.cpu_percent(interval=None, percpu=True)))

        if success:
            print(f"  [SUCCESS] Produced engine for '{model_name}'")
        else:
            print(f"  [ERROR] Failed to warm '{model_name}': {err}")

    # Final CPU report
    print("-" * 30)
    print(f"Warming process complete.")
    print(f"Peak per-core utilization detected: {peak_usage}%")

if __name__ == "__main__":
    main()
