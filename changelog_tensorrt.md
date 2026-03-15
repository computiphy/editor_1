# TensorRT Branch Changelog

This document tracks the development, optimization, and debugging of TensorRT integration within the Wedding AI Pipeline.

## 2026-03-10 to 2026-03-15: TensorRT Infrastructure

### Core Integration
- **New Device Support:** Added `"tensorrt"` as a valid `device` option in `BackgroundRemovalConfig` (`src/config/schema.py`).
- **Provider Cascade:** Implemented a prioritized execution provider list in `BackgroundRemover` (`src/segmentation/background_remover.py`):
  1. `TensorrtExecutionProvider` (Primary)
  2. `CUDAExecutionProvider` (Fallback)
  3. `CPUExecutionProvider` (Hard Fallback)
- **Engine Caching:** Implemented centralized TensorRT engine caching. Engines are now compiled once and saved to `.trt_engine_cache/` in the project root to eliminate the 10-30s graph analysis overhead on subsequent runs.
- **Environment Tuning:** Maintained `OMP_NUM_THREADS="1"` for TensorRT to prevent CPU core starvation of the parallel color grading workers.

### Windows Debugging and DLL Dependency Mapping
- **Issue:** Encountered `Error 126` (Module not found) despite TensorRT being in the System PATH.
- **Findings:** 
  - Identified that even with TensorRT present, `nvinfer_10.dll` fails to load if intermediate dependencies (`cublas64_12.dll`, `cudnn64_9.dll`) are only present in a Python virtual environment via pip.
  - Documented the requirement for system-level installation of CUDA and cuDNN to ensure the Windows DLL loader can follow the dependency chain.
  - Corrected documentation to point to the `bin/` directory rather than `lib/` for PATH configuration.

### Bug Fixes and UX Improvements (Commit: 979d7c2)
- **Dynamic Stage Labeling:** Refactored `orchestrator.py` to use context-aware labels (e.g., "Restoration", "Caching", "Grading") based on active configuration flags. This prevents misleading "Grading" logs when the feature is disabled.
- **UnboundLocalError Fix:** Resolved a crash in the dynamic labeling logic where `bg_remover` was referenced before initialization. Fixed via TDD workflow.

### Hardware and Environment Automation (Commit: ca1a879)
- **Windows DLL Auto-Discovery:** Implemented a runtime discovery shim in `BackgroundRemover` that automatically scans for NVIDIA/TensorRT DLLs in `C:\Program Files\NVIDIA` and `C:\Program Files\NVIDIA GPU Computing Toolkit`. This eliminates the "Missing DLL" errors without requiring manual System PATH configuration.
- **Venv Integration:** Added automatic registration of pip-installed NVIDIA binaries inside the virtual environment as a secondary search layer.

### Documentation and Safety
- **README.md:** Updated with specific `pip install onnxruntime-gpu` instructions and clear warnings about Windows-specific binary requirements for TensorRT.
- **Hardware Scaling:** Documented the "VRAM Inflation" behavior, where 1GB models expand to 12GB+ during inference due to activation maps.
- **Git Hygiene:** Added `.trt_engine_cache/` to `.gitignore` to prevent multi-gigabyte binary engines from bloating the repository.

### Diagnostic Scripts (Developed and Retired)
- `test_dll.py`: Primitive `ctypes` loader to check `PATH` visibility.
- `check_deps.py`: PE-file analyzer to map out missing DLL links in the `nvinfer` chain.
- `test_dll_venv.py`: Attempted to bridge venv-installed CUDA with system-installed TensorRT.
