# Changelog

## `01b2f0f` (Current)
**Date:** 2026-03-10  
**Summary:** Hardware profiling, GIL contention discovery, and optimal thread tuning.

---

### 🔍 Performance: Eliminating Contention Bottlenecks

Extensive profiling was conducted to resolve a performance regression (200s → 370s) introduced by the multi-threaded architecture. The findings fundamentally changed how we scale the pipeline on high-end hardware (e.g., RTX 3070 Ti, 48GB RAM).

#### 1. The Python GIL (Global Interpreter Lock) Bottleneck
- **Discovery:** Profiling 10 grading workers revealed **95+ seconds** of pure thread lock contention (`_thread.lock.acquire`) and saturated RAM bandwidth via `numpy.clip`.
- **Resolution:** Reduced `color_grading.workers` to `3`. This optimal number eliminates lock contention and allows OpenCV/Numpy to drop the GIL efficiently, restoring CPU performance without saturating the DDR bus.

#### 2. ONNX Runtime CUDA Contention
- **Discovery:** Profiling 2 GPU workers (`workers_gpu: 2`) revealed severe CUDA context-switching logic inside `onnxruntime-gpu`. Two threads fighting for the same GPU session doubled inference time per image (from ~12s to ~30s+) despite abundant VRAM on the 3070 Ti.
- **Resolution:** Kept `background_removal.workers` at `1`. The producer-consumer queue ensures this single GPU thread is kept at 100% utilization by the grading stage, maximizing throughput without context-switching overhead.

#### [color_only_config.yaml](file:///C:/Swaroop/editor_1/editor_1/configs/color_only_config.yaml)
- Updated defaults to globally optimal values to prevent user hardware saturation:
  - `workers_grading: 3` (GIL sweet-spot)
  - `workers_gpu: 1` (ONNX CUDA sweet-spot)
  - `queue_maxsize: 8`

---
## `8f8150f` ← `fa2cd5f`
**Date:** 2026-03-10  
**Summary:** Performance optimizations, multithreaded pipeline architecture, and hardware scaling documentation.

---

### 🏗️ Architecture: Stage-Specific ThreadPools with Producer-Consumer Queue

The orchestrator was completely refactored from a sequential single-threaded loop to a multi-stage concurrent architecture.

#### [orchestrator.py](file:///C:/Swaroop/editor_1/editor_1/src/pipeline/orchestrator.py)

- **Culling stage parallelized:** Replaced the sequential `for p in tqdm(photo_paths)` loop with `ThreadPoolExecutor(max_workers=workers_culling)`. Each image is now evaluated concurrently.
- **Grading/Restoration stage (Producer):** Extracted into `_grade_image()` function running in `ThreadPoolExecutor(max_workers=workers_grading)`. Loads images, applies restoration, color grading, LUT, saves to `final/`, and pushes the graded array into a bounded `queue.Queue`.
- **Background Removal stage (Consumer):** Extracted into `_bg_consumer()` running in `ThreadPoolExecutor(max_workers=workers_gpu)`. Continuously pulls graded images from the queue and runs `rembg` inference. Uses a **sentinel shutdown pattern** (`None` values pushed after producers finish) to cleanly terminate consumer threads without deadlocks.
- **SemanticSegmenter race condition fix:** Pre-initialized before any thread pool starts, eliminating the unsafe `if not hasattr(self, '_segmenter')` lazy-init that previously risked duplicate instantiation across threads.
- **Removed:** The old monolithic `_process_single_image()` function.

---

### ⚡ Performance: uint8 Optimizations

#### [semantic_segmenter.py](file:///C:/Swaroop/editor_1/editor_1/src/segmentation/semantic_segmenter.py)

- Refactored `segment()` to perform boolean mask operations on `uint8` arrays instead of `float32`. Deferred `float32` casting until after morphology and only for final masks before blurring. Significantly reduces CPU load and memory allocation per image.

#### [engine.py (ColorGradingEngine)](file:///C:/Swaroop/editor_1/editor_1/src/color/engine.py)

- Optimized `_override_skin`, `_override_sky`, `_override_vegetation`, `_override_white_dress`, `_override_dark_suit` to operate on `uint8` arrays where possible, reducing redundant `float32` conversions.
- Optimized `_apply_per_channel_hsl` to use `uint8` matrices and avoid unnecessary full-scale float computations.

#### [orchestrator.py](file:///C:/Swaroop/editor_1/editor_1/src/pipeline/orchestrator.py)

- Replaced PIL `Image.save()` PNG export with `cv2.imwrite()` using `cv2.IMWRITE_PNG_COMPRESSION, 1` for significantly faster I/O.
- Fixed `UnboundLocalError` caused by a redundant local `import cv2` inside the processing loop.

---

### 🔒 Thread-Safety: Background Remover Locking

#### [background_remover.py](file:///C:/Swaroop/editor_1/editor_1/src/segmentation/background_remover.py)

- Added `threading.Lock()` to `BackgroundRemover.__init__()`.
- Wrapped `_get_session()` with a **double-checked locking pattern** to prevent multiple threads from simultaneously initializing the ONNX Runtime session on first use.
- Changed `post_process_mask` default to `False` (redundant CPU-heavy post-processing).
- Added `cudnn_conv_algo_search: "HEURISTIC"` and `arena_extend_strategy: "kNextPowerOfTwo"` to `CUDAExecutionProvider` options for faster GPU warm-up.

---

### 📐 Config: Stage-Specific Worker Counts

#### [schema.py](file:///C:/Swaroop/editor_1/editor_1/src/config/schema.py)

Added new fields to `PipelineConfig` (backward-compatible — old configs with only `workers` still work):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `workers_culling` | `Optional[int]` | `None` (→ `workers`) | Thread count for culling stage |
| `workers_grading` | `Optional[int]` | `None` (→ `workers`) | Thread count for grading/restoration |
| `workers_gpu` | `Optional[int]` | `None` (→ `1`) | Thread count for GPU inference (BG removal) |
| `queue_maxsize` | `int` | `4` | Max items in the producer-consumer RAM queue |

#### [color_only_config.yaml](file:///C:/Swaroop/editor_1/editor_1/configs/color_only_config.yaml)

Updated with stage-specific worker counts:
```yaml
workers: 2              # Legacy fallback
workers_culling: 12     # CPU/Disk bound
workers_grading: 10     # CPU/RAM bound
workers_gpu: 2          # VRAM bound
queue_maxsize: 4        # ~400MB RAM buffer
```

---

### 📝 Documentation

#### [NEW] [hardware_scaling.md](file:///C:/Swaroop/editor_1/editor_1/hardware_scaling.md)

- Documents how 11 hardware parameters (CPU single-thread, AVX-512, core count, RAM bandwidth/latency/capacity, GPU VRAM/compute/bandwidth/generation, storage speed) impact each pipeline stage.
- Includes a 0-10 impact matrix mapping every hardware parameter to every stage.
- Includes execution time estimates for two reference hardware configurations (GTX 1650 4GB vs RTX 3070 Ti 8GB).

#### [.gitignore](file:///C:/Swaroop/editor_1/editor_1/.gitignore)

- Added `.prof` profiling output files and `full_stats.txt` to prevent profiling artifacts from being committed.

---

### 🧪 Tests

#### [test_pipeline.py](file:///C:/Swaroop/editor_1/editor_1/tests/test_pipeline.py)

| Test | Status | Description |
|---|---|---|
| `test_pipeline_stage_specific_workers` | ✅ NEW | Validates stage-specific worker config fields are accepted |
| `test_pipeline_producer_consumer_queue` | ✅ NEW | Validates bounded queue with mocked BG removal, sentinel shutdown, and cutout output |
| `test_pipeline_multithreading` | ❌ REMOVED | Replaced by the two tests above |

#### [test.md](file:///C:/Swaroop/editor_1/editor_1/test.md)

- Updated the test registry to reflect the new test names and rationale.

---

### Full File Change Summary

| File | Insertions | Deletions | Net |
|---|---|---|---|
| `src/pipeline/orchestrator.py` | +195 | -93 | +102 |
| `tests/test_pipeline.py` | +96 | -36 | +60 |
| `hardware_scaling.md` | +118 | 0 | +118 (NEW) |
| `src/color/engine.py` | +68 | -68 | 0 |
| `src/segmentation/semantic_segmenter.py` | +24 | -24 | 0 |
| `src/segmentation/background_remover.py` | +14 | -7 | +7 |
| `src/config/schema.py` | +5 | -1 | +4 |
| `configs/color_only_config.yaml` | +14 | -7 | +7 |
| `test.md` | +2 | -1 | +1 |
| `.gitignore` | +6 | 0 | +6 |
| **Total** | **+441** | **-138** | **+303** |
