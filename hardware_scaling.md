# Hardware Scaling Impact

This document details how specific hardware component upgrades affect the overall performance of the Wedding AI Pipeline, broken down by processing stages.

## Processing Stages Overview
- **Stage 1 (I/O & Culling):** Reading massive JPEGs, blur detection (Laplacian variance), perceptual hashing, BRISQUE/NIQE scoring.
- **Stage 2 (Restoration):** NAFNet/GFPGAN tile-based upscaling and noise reduction.
- **Stage 3 (Color Grading & LUTs):** CPU-bound `float32` matrix math for Oklab conversions, contrast curves, frequency separation, and tetrahedral 3D LUT blending.
- **Stage 4 (Segmentation & BG Removal):** ONNX Runtime (`birefnet-portrait`) generating segmentation masks.
- **Stage 5 (Save/Export):** Compressing multi-megapixel numpy arrays back into JPEGs and RGBA PNGs.

---

### 1. CPU Single Thread Performance
*   **What it is:** How fast a single core can execute instructions (IPC and clock speed).
*   **Impact on Pipeline:**
    *   **I/O & Export:** Highly critical. Python's `cv2.imdecode` and `cv2.imencode` (jpeg compression algorithms) are primarily single-threaded bounds per image. Faster single-core performance directly drops decode/encode latency.
    *   **Color Grading:** Critical for spline curve interpolation and sequential array manipulations where Python's GIL prevents perfect parallelization inside a single image's grading loop.
    *   **Orchestrator Overhead:** High single-thread speed reduces the micro-stutter when Python passes data between thread pools.

### 2. AVX-512 Support
*   **What it is:** Advanced Vector Extensions that allow a CPU to process 512 bits of data in a single clock cycle.
*   **Impact on Pipeline:**
    *   **Color Grading:** Massive impact. NumPy is compiled against math backends (like OpenBLAS/MKL) that heavily utilize AVX-512 for large floating-point matrix multiplications. Converting a 24-megapixel image between sRGB and Oklab requires multiplying millions of pixels against 3x3 matrices; AVX-512 can mathematically double or quadruple the throughput here compared to standard AVX2.

### 3. CPU Core Count
*   **What it is:** The number of physical processing units available.
*   **Impact on Pipeline:**
    *   **Culling:** Scales nearly linearly. If you double your cores, you can hash and blur-detect double the images per second using the `workers_culling` thread pool.
    *   **Color Grading:** Scales very well. Each core can independently grade a full image via the CPU thread pool. Moving from 4 cores to 8 cores will almost halve grading time entirely, as long as RAM bandwidth doesn't choke.
    *   **GPU Stages:** Minimal direct impact, but having extra cores prevents the system from stuttering while the GPU is working.

### 4. RAM Bandwidth
*   **What it is:** The speed at which data can be shuttled from memory to the CPU (e.g., DDR4-2400 vs DDR5-6000).
*   **Impact on Pipeline:**
    *   **Color Grading & Culling:** Extreme impact. The pipeline creates massive temporary NumPy arrays (96MB+ per `float32` image map). The grading engine constantly forces the CPU to re-read these mega-arrays to apply layers (CLAHE, Curves, LUT). Slow RAM bandwidth physically starves a fast CPU of data, meaning 8-core CPUs spend most of their time waiting for the DDR4 bus instead of doing math. DDR5 dramatically unlocks CPU-bound image grading.

### 5. RAM Latency
*   **What it is:** The delay before memory can begin sending data (CAS Latency).
*   **Impact on Pipeline:**
    *   **Overall:** Minimal impact. Image processing is a streaming workload with huge sequential blocks of data. Bandwidth matters far more than the nanosecond start-delay latency for big NumPy operations.

### 6. RAM Capacity
*   **What it is:** The total available memory (e.g., 16GB vs 48GB).
*   **Impact on Pipeline:**
    *   **Queue Sizes & Parallelism:** Determines how deep the Producer/Consumer Queue can run. With 16GB, you can only safely process 4 hi-res graded images in RAM simultaneously before paging to the SSD (which causes catastrophic slowdowns). With 48GB, you can safely hold 16+ graded arrays in RAM, effectively decoupling CPU and GPU synchronization and ensuring 100% component utilization.

### 7. GPU VRAM Capacity
*   **What it is:** The memory physically on the graphics card (e.g., 4GB vs 8GB vs 24GB).
*   **Impact on Pipeline:**
    *   **Restoration & BG Removal:** Forms the hard ceiling for AI inferences. The `birefnet-portrait` model dynamically scales memory based on resolution. A 4GB GPU can only process one 24MP image at a time natively (or requires extreme tiling). An 8GB GPU allows `workers_gpu: 3`, meaning 3 photos are simultaneously masked on the card.
    *   *Note:* Running out of VRAM terminates the pipeline entirely with `CUDA_OUT_OF_MEMORY`.

### 8. GPU Compute Power (CUDA Cores & Clock Speed)
*   **What it is:** The raw number-crunching muscle of the graphics card.
*   **Impact on Pipeline:**
    *   **Restoration & BG Removal:** Determines how many milliseconds it takes to push a tensor through the AI model. Upgrading from a GTX 1650 to an RTX 4070 easily yields a 400% - 600% reduction in the inference latency per image for the `rembg` stage.

### 9. GPU Memory Bandwidth
*   **What it is:** The speed at which the GPU core can read its own VRAM (e.g., GDDR6 vs GDDR6X).
*   **Impact on Pipeline:**
    *   **Restoration & BG Removal:** High impact for high-resolution images. NAFNet and BiRefNet models require heavy convolution operations over massive image tensors. A GPU with a wider, faster memory bus will process these tensors much faster, preventing the CUDA cores from starving.

### 10. GPU Generation (Ampere, Lovelace, Blackwell)
*   **What it is:** The underlying architecture of the GPU chip.
*   **Impact on Pipeline:**
    *   **AI Inference:** Generational jumps bring vast architectural improvements to Tensor Cores (specialized hardware for AI matrix multiplication). An RTX 4060 (Lovelace) will often dramatically outperform an RTX 3060 (Ampere) in ONNX/PyTorch execution specifically because the Lovelace tensor cores are significantly more efficient at FP16/FP32 math, even if raw total CUDA core counts look similar.

### 11. Storage Speed (HDD, SATA SSD, NVMe Gen 3/4/5)
*   **What it is:** Disk read/write throughput.
*   **Impact on Pipeline:**
    *   **I/O & Culling:** A 24MP RAW or strict JPEG might be 15MB-30MB. A directory of 2,000 photos is 60GB of data. 
    *   *HDD (~100 MB/s):* Catastrophically bad. Thrashing read-heads will bottleneck the culling pool instantly.
    *   *SATA SSD (~500 MB/s):* Good for sequential, but bottlenecks under heavy multi-threaded random reads/writes.
    *   *NVMe Gen 3/4/5 (3000 -> 12000 MB/s):* Crucial for keeping the pipeline fed. Writing massive uncompressed cutouts to the SSD instantly clears RAM queues. A Gen 4 NVMe allows the export threads to write out 24MP PNGs invisibly without blocking the orchestrator.

## Hardware Impact Matrix (0-10)

Scale: `0` (Nil / Almost Nil) to `10` (Extremely High Impact)

| Hardware Parameter | Stage 1 (Culling) | Stage 2 (Restoration) | Stage 3 (Color Grading) | Stage 4 (BG Removal) | Stage 5 (Export) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU Single Thread** | 3 | 1 | 9 | 1 | 10 |
| **AVX-512 Support** | 1 | 0 | 10 | 0 | 2 |
| **CPU Core Count** | 9 | 1 | 8 | 1 | 8 |
| **RAM Bandwidth** | 7 | 2 | 10 | 3 | 8 |
| **RAM Latency** | 2 | 1 | 3 | 1 | 2 |
| **RAM Capacity** | 4 | 5 | 9 | 7 | 4 |
| **GPU VRAM Capacity** | 0 | 10 | 0 | 10 | 0 |
| **GPU Compute Power** | 0 | 10 | 0 | 10 | 0 |
| **GPU Mem Bandwidth** | 0 | 9 | 0 | 9 | 0 |
| **GPU Generation** | 0 | 8 | 0 | 8 | 0 |
| **Storage Speed** | 10 | 1 | 2 | 1 | 10 |

## Execution Time Estimates (Config 1 vs Config 2)

Assuming a typical processing batch of **100 high-resolution (24MP) images** with the full pipeline enabled (Culling, Grading, and Background Removal).

### Config 1 
*(4 Core Zen+, 16GB RAM, Gen3 NVMe, GTX 1650 4GB)*
**Queue Setup:** `workers_culling: 6`, `workers_grading: 4`, `workers_gpu: 1`, `queue_maxsize: 4`

*   **Stage 1 (Culling):** ~15 seconds (Bottlenecked by CPU math and average Gen3 NVMe read speeds).
*   **Stage 3 (Color Grading):** ~150 seconds (1.5s per image across 4 limited cores).
*   **Stage 2 + 4 (Restoration & BG Removal):** ~1,200 seconds (12s per image on average. The GTX 1650 is severely bottlenecked on tensor math, and the 4GB VRAM hard-limits the pipeline to 1 concurrent ONNX worker).
*   **Total Exec Time:** **~21 minutes**
*   *Note:* The entire pipeline runs effectively at the speed of the GPU step, as the CPU queue (`maxsize=4`) fills immediately and idles while waiting for the GTX 1650.

### Config 2
*(8 Core Zen 3+, 48GB Fast DDR5 RAM, Gen4 NVMe, RTX 3070 Ti 8GB)*
**Queue Setup:** `workers_culling: 12`, `workers_grading: 10`, `workers_gpu: 3`, `queue_maxsize: 16`

*   **Stage 1 (Culling):** ~5 seconds (8 fast physical cores and Gen4 NVMe simply tear through the file headers).
*   **Stage 3 (Color Grading):** ~30 seconds (0.3s effective per image across 10 threads, powered by immense AVX-supported DDR5 throughput).
*   **Stage 2 + 4 (Restoration & BG Removal):** ~150 seconds (The RTX 3070 Ti takes roughly 4.5s per image natively. Since we run 3 workers concurrently across the 8GB VRAM pool, the effective processing rate is 1.5s per image).
*   **Total Exec Time:** **~2.5 minutes**
*   *Note:* The upgrade provides an incredible **~8x to ~10x total speedup**, driven entirely by unlocking the RAM bandwidth bottleneck during grading, and utilizing the massive 16-item RAM queue to keep 3 concurrent GPU inferences fed at 100% saturation.
