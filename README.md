# Wedding Photography AI Pipeline

**Local-first, open-source AI-powered pipeline for automating wedding photography post-production.**

A modular, configuration-driven system that takes raw wedding photos and produces professionally graded, restored, and export-ready images — all running on your own hardware with no cloud dependencies.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Stages](#pipeline-stages)
  - [1. Ingestion](#1-ingestion)
  - [2. Intelligent Culling](#2-intelligent-culling)
  - [3. AI Restoration](#3-ai-restoration)
  - [4. Color Grading (AI Colorist)](#4-color-grading-ai-colorist)
  - [5. Background Removal](#5-background-removal)
  - [6. Smart Cropping](#6-smart-cropping)
  - [7. Watermarking](#7-watermarking)
  - [8. Narrative Grouping](#8-narrative-grouping)
  - [9. Album Layout](#9-album-layout)
  - [10. Reporting](#10-reporting)
- [Configuration System](#configuration-system)
- [Filter Presets Reference](#filter-presets-reference)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)

---

## Architecture Overview

The pipeline follows a **micro-tool architecture** with SOLID principles. Each processing stage is an independent engine with its own module, tests, and configuration. The orchestrator (`src/pipeline/orchestrator.py`) chains them together based on the YAML config, so every stage can be enabled or disabled independently.

```
Input Photos (RAW/JPEG)
    │
    ├── 1. INGESTION ──────────── Glob + dedup
    ├── 2. CULLING ────────────── Blur / Face / Quality / Duplicate detection
    ├── 3. RESTORATION ────────── NAFNet (deblur) + GFPGAN (face restore)
    ├── 4. COLOR GRADING ──────── 19 filter presets + per-channel HSL + split toning
    ├── 5. BACKGROUND REMOVAL ── BiRefNet-Portrait → RGBA PNG cutouts
    ├── 6. CROPPING ───────────── RT-DETR subject-aware crop (1:1, 9:16)
    ├── 7. WATERMARK ──────────── Auto-positioned logo overlay
    ├── 8. NARRATIVE ──────────── CLIP + DBSCAN scene clustering
    ├── 9. LAYOUT ─────────────── Album page generation
    └── 10. REPORTING ─────────── JSON report + PDF summary
```

**Key Design Decisions:**
- **Configuration-driven:** Every stage has `enabled: true/false` — run only what you need.
- **GPU-aware:** Auto-detects CUDA/CPU. VRAM tiling for GPUs with <6GB VRAM (e.g., GTX 1650).
- **No cloud dependencies:** All models run locally. No API keys, no uploads.
- **TDD:** 18 test files with comprehensive coverage across all modules.

---

## Pipeline Stages

### 1. Ingestion

**Module:** `src/pipeline/orchestrator.py` (inline)
**What it does:** Scans the input directory for all supported image formats and deduplicates them.

| Detail | Value |
|:-------|:------|
| Supported Formats | `RAW`, `JPG`, `JPEG` (configurable) |
| Deduplication | Uses `set()` on resolved `Path` objects to prevent case-insensitive duplicates on Windows (e.g., `photo.jpg` and `photo.JPG` are the same file) |
| Sorting | Alphabetical by filename for deterministic processing order |

**Why this approach:** Windows filesystems are case-insensitive, so globbing for both `*.jpg` and `*.JPG` can produce duplicates. We use a set-based approach to guarantee each file is processed exactly once.

---

### 2. Intelligent Culling

**Module:** `src/culling/`
**What it does:** Automatically identifies and removes low-quality photos before expensive processing stages.

#### Sub-engines:

| Engine | File | Method | Purpose |
|:-------|:-----|:-------|:--------|
| **Blur Detector** | `blur_detector.py` | Laplacian variance (FFT energy) | Detects motion blur and out-of-focus shots |
| **Face Analyzer** | `face_analyzer.py` | MediaPipe Face Mesh | Detects faces, blink detection, expression scoring |
| **Quality Assessor** | `quality_assessor.py` | BRISQUE / NIQE metrics | No-reference image quality assessment |
| **Duplicate Clusterer** | `duplicate_cluster.py` | Perceptual hashing (ImageHash) + DBSCAN | Groups near-duplicate burst shots |

**Library choices:**

- **MediaPipe** (face analysis) — *Selected over* InsightFace and dlib.
  - *Why:* Runs on CPU with excellent performance. InsightFace requires complex ONNX/CUDA setup that frequently fails on Windows. dlib's face detection is slower and less accurate on modern photos. MediaPipe provides 468 facial landmarks, blink detection, and expression analysis out of the box.

- **ImageHash** (duplicate detection) — *Selected over* SSIM and raw pixel comparison.
  - *Why:* Perceptual hashing (pHash) is O(1) comparison after initial hash computation, making it viable for 1000+ image sets. SSIM requires loading both images into memory for each pair comparison (O(n²) memory). pHash is robust to minor exposure changes between burst shots.

- **scikit-learn DBSCAN** (clustering) — *Selected over* K-means and agglomerative clustering.
  - *Why:* DBSCAN doesn't require a pre-specified number of clusters (we don't know how many burst groups exist). It naturally handles noise (standalone photos that aren't part of any group).

**Config:**
```yaml
culling:
  enabled: true
  blur_threshold: 100.0       # Laplacian variance below this = blurry
  duplicate_threshold: 5      # Hamming distance for perceptual hash match
  quality_threshold: 70.0     # BRISQUE score cutoff
```

---

### 3. AI Restoration

**Module:** `src/restoration/`
**What it does:** Enhances image quality by deblurring, denoising, and restoring facial details.

#### Sub-engines:

| Engine | File | Model | Purpose |
|:-------|:-----|:------|:--------|
| **NAFNet Restorer** | `nafnet_restore.py` | NAFNet-DeBlur | General image deblurring and denoising |
| **GFPGAN Restorer** | `gfpgan_restore.py` | GFP-GAN v1.4 | Face-specific restoration (skin, eyes, teeth) |
| **Restoration Router** | `engine.py` | Rule-based | Routes to NAFNet or GFPGAN based on whether faces are detected and blur severity |

**VRAM Tiling System:**

The pipeline is designed for low-VRAM GPUs (like the GTX 1650 with 4GB). The `ImageTiler` utility (`src/utils/tiling.py`) splits images into overlapping 512×512 tiles, processes each tile through the model individually, and stitches them back together with blending in the overlap region.

```
┌──────────────────────────┐
│     Original Image       │
│ ┌────┬────┬────┐         │
│ │ T1 │ T2 │ T3 │←512px   │
│ ├────┼────┼────┤         │
│ │ T4 │ T5 │ T6 │  ↕64px  │
│ └────┴────┴────┘ overlap │
└──────────────────────────┘
```

**Library choices:**

- **NAFNet** (deblurring) — *Selected over* Real-ESRGAN and DeblurGAN-v2.
  - *Why:* NAFNet achieves state-of-the-art PSNR on GoPro/REDS deblur benchmarks while being significantly lighter than Real-ESRGAN (which is primarily a super-resolution model and uses excessive VRAM). NAFNet's "Nonlinear Activation Free" design makes it faster at inference.

- **GFPGAN** (face restoration) — *Selected over* CodeFormer and RestoreFormer.
  - *Why:* GFPGAN produces the most natural-looking face restorations in our testing. CodeFormer tends to over-smooth skin texture. GFPGAN's GAN-based approach preserves pore-level detail while fixing degradation.

- **PyTorch** (inference backend) — *Selected over* ONNX Runtime and TensorFlow.
  - *Why:* Native CUDA support, the widest ecosystem of pretrained model weights, and the simplest integration path. Most state-of-the-art vision models are published as PyTorch checkpoints first.

**Config:**
```yaml
restoration:
  enabled: true
  auto_route: true           # Auto-select NAFNet vs GFPGAN based on content
  primary_model: "nafnet"
  face_restore: true
  tiling:
    enabled: true            # REQUIRED for GPUs < 6GB VRAM
    tile_size: 512
    overlap: 64
```

---

### 4. Color Grading (AI Colorist)

**Module:** `src/color/engine.py` (Legacy V1) and `src/color/engine_v2.py` (SOTA V2)

The Wedding AI Pipeline features a professional-grade color engine designed to compete with high-end desktop software like Capture One and DaVinci Resolve.

#### 🚀 SOTA Color Engine V2
The V2 engine represents the pinnacle of digital color science, moving away from standard RGB/HSV math to a physically accurate, perceptually uniform pipeline.

| Feature | Technical Implementation | Rationale |
|:--------|:-------------------------|:----------|
| **Color Space** | **Oklab** (Internal Working Space) | Perceptually uniform. Hue rotations do not cause unwanted brightness or saturation shifts. |
| **Bit Depth** | **32-bit Float** (End-to-end) | Eliminates rounding errors and "banding" common in 8-bit uint8 pipelines. |
| **Working Space** | **ACEScg** (Scene-Linear) | Industry-standard wide gamut space used in Hollywood VFX. Preserves highlight detail and color accuracy. |
| **Tone Curves** | **Cubic Spline** (4096-point) | Replaces simple gamma with smooth, precise contrast control across the entire luminance range. |
| **Saturation** | **Subtractive (Filmic)** | Emulates physical film dyes. Colors become *denser* and *richer* as they saturate, rather than just "brighter." |
| **3D LUTs** | **Tetrahedral Interpolation** | Supports `.cube` files with the most precise interpolation method available (superior to trilinear). |
| **Grain** | **Luminance-Mapped Perlin** | Spatially correlated "clumpy" grain that varies by brightness — absence in highlights, rich in shadows. |
| **Effects** | **Halation & Red-Scatter** | Emulates the red glowing edge around highlights seen in traditional film stocks. |

#### Processing Pipeline — SOTA V2:
```
 1. Input Norm ─────── uint8 → float32 [0-1]
 2. CAT02 WB ───────── Chromatic adaptation (physically correct white balance)
 3. ACES Transform ─── Into ACEScg Wide-Gamut Linear Space
 4. Exposure ───────── Mathematically correct gain in linear space
 5. Oklab ──────────── Perceptually uniform working space
 6. Tone Curve ─────── Cubic Spline (4096-pt LUT) applied to L channel
 7. Contrast ───────── S-curve in Oklab L
 8. Per-Channel HSL ── Target specific hues (e.g., Greens → Emerald)
 9. Split Toning ───── Colorize shadows and highlights independently
10. Filmic Sat ─────── Subtractive saturation boost (adds density)
11. 3D LUT ─────────── Tetrahedral .cube application (Optional)
12. Halation ───────── Highlight red-channel scattering (Bloom)
13. Vignette ───────── Radial falloff
14. Perlin Grain ───── Luminance-weighted procedural texture
15. Output ─────────── Final sRGB Gamma encode → uint8
```

#### Available Presets (19 total):

| Preset | Style | Best For |
|:-------|:------|:---------|
| `natural` | Clean, true-to-life | Documentary style |
| `cinematic` | Teal shadows / Orange highlights | Dramatic receptions |
| `pastel` | Soft, desaturated, dreamy | Bridal prep, flat-lays |
| `moody` | Dark, dramatic, editorial | Evening, artistic portraits |
| `golden_hour` | Warm amber tones | Late afternoon outdoor shots |
| `film_kodak` | Kodak Portra 400 emulation | Classic wedding look |
| `film_fuji` | Fujifilm Pro 400H emulation | Cool-toned film look |
| `vibrant` | Punchy, Instagram-ready | Colorful celebrations |
| `black_and_white` | Rich monochrome | Timeless elegance |
| `moody_forest` | Cool greens, dark tones | Forest/garden venues |
| `golden_hour_portrait` | Warm, skin-focused | Couple portraits |
| `urban_cyberpunk` | Neon purple/cyan | City/urban shoots |
| `vintage_painterly` | Faded, low dynamic range | Artistic/editorial |
| `high_fashion` | Sharp, bold, magazine | Fashion-forward weddings |
| `sepia_monochrome` | Warm-toned B&W | Heritage/classic style |
| `vibrant_landscape` | Saturated, punchy | Scenic venue shots |
| `lavender_dream` | Pink/purple, ethereal | Romantic/fantasy |
| `bleach_bypass` | Gritty, desaturated, high contrast | Edgy editorial |
| `dark_academic` | Green/orange, dim, intellectual | Library/manor venues |

#### Strength Control:
Every preset's adjustments are scaled by the `strength` parameter (0.0 = no effect, 1.0 = full, 1.5 = extreme), allowing fine-tuning the intensity of any look.

#### Optional Reference Transfer:
If a `reference_image` path is provided, the engine performs a **Reinhard et al. (2001) Lab-space statistical transfer** to match the color statistics of a "hero shot," blended at 30% with the preset result.

**Library choices:**

- **Oklab (pure NumPy)** — *Selected over* OpenCV HSV/LAB and colour-science CIELAB.
  - *Why:* Oklab is perceptually uniform (unlike CIELAB) and can be implemented in pure NumPy with zero external dependencies. Hue rotations don't cause luminance shifts. Saturation adjustments don't cause hue drift. This eliminates the two most common artifacts in wedding photo grading.

- **SciPy CubicSpline** (tone curves) — *Selected over* numpy LUT and manual gamma formulas.
  - *Why:* Cubic splines allow arbitrary multi-point curves (like DaVinci Resolve), enabling precise control over shadow/midtone/highlight response. The spline is pre-computed into a 4096-point float32 LUT, so per-pixel application is still O(1).

- **Subtractive Saturation** — *Custom implementation* based on film density science.
  - *Why:* This is the single biggest quality differentiator. Every consumer photo editor uses additive saturation (making colors brighter). Our engine uses subtractive saturation (making colors denser), producing professional results indistinguishable from real film stock.

**Config:**
```yaml
color_grading:
  enabled: true
  style: "cinematic"              # Any of the 19 presets
  strength: 1.0                   # 0.0 to 1.5
  reference_image: "path/to/hero_shot.jpg"  # Optional
  segmentation_enabled: false     # SAM-based local grading (future)
```

---

### 5. Background Removal

**Module:** `src/segmentation/background_remover.py`
**What it does:** Removes the background from each photo, producing RGBA PNG images with transparent backgrounds. The color grading preset is applied *before* background removal, so cutouts inherit the selected look.

**Output:** Saved as PNG files in a `cutouts/` directory alongside the `final/` graded images.

#### Available Models:

| Model | Quality | Speed | Best For |
|:------|:--------|:------|:---------|
| **`birefnet-portrait`** ⭐ | Excellent | ~55s/img | **People, weddings, portraits** |
| `birefnet-massive` | Highest | ~90s/img | Any scene, maximum accuracy |
| `birefnet-general` | Very Good | ~45s/img | General-purpose |
| `bria-rmbg` | Good | ~15s/img | Fast commercial-grade |
| `u2net` | Fair | ~14s/img | Simple objects only |

**Library choices:**

- **rembg + BiRefNet-Portrait** — *Selected over* SAM (Segment Anything), MODNet, and raw U²-Net.
  - *Why:* SAM requires manual prompts (point/box) and produces class-agnostic masks — it doesn't understand "person" vs "background." MODNet is portrait-specific but hasn't been updated since 2021 and struggles with complex poses. BiRefNet-Portrait is specifically trained on human segmentation datasets with SOTA accuracy on portrait benchmarks, handles hair/veils/flowing fabric well, and integrates cleanly via `rembg`.

- **Alpha Edge Refinement** — A 3px Gaussian blur is applied to the alpha channel after segmentation, softening jagged edges for more natural compositing.

**Config:**
```yaml
background_removal:
  enabled: true                       # true / false
  model: "birefnet-portrait"          # See model table above
```

---

### 6. Smart Cropping

**Module:** `src/cropping/engine.py`
**What it does:** Generates social media-ready crops (1:1 for Instagram, 9:16 for Stories/Reels) with subject-aware composition.

**Library choices:**

- **RT-DETR** (subject detection) — *Selected over* YOLOv8 and Faster R-CNN.
  - *Why:* RT-DETR (Real-Time DEtection Transformer) is the first real-time end-to-end object detector, eliminating the need for NMS post-processing. It's more accurate than YOLOv8 on COCO benchmarks while being similarly fast. For wedding photos, accurate person detection is critical for centering the crop.

**Config:**
```yaml
cropping:
  enabled: true
  ratios: ["1:1", "9:16"]
  detector: "rtdetr"
```

---

### 7. Watermarking

**Module:** `src/watermark/engine.py`
**What it does:** Overlays a semi-transparent watermark logo on output images.

Features:
- Auto-positioning (avoids placement over detected faces)
- Configurable opacity
- Supports PNG watermarks with transparency

**Config:**
```yaml
watermark:
  enabled: true
  path: "assets/watermark.png"
  position: "auto"
  opacity: 0.5
```

---

### 8. Narrative Grouping

**Module:** `src/narrative/engine.py`
**What it does:** Groups photos into "chapters" or "scenes" based on visual similarity using AI embeddings.

**Library choices:**

- **OpenCLIP** (embeddings) — *Selected over* ResNet features and raw pixel histograms.
  - *Why:* CLIP embeddings capture semantic meaning ("couple at altar" vs "couple at reception") rather than just low-level visual similarity. Two photos of the same scene from different angles will cluster together because CLIP understands content, not just pixel patterns.

- **DBSCAN** (clustering) — Same rationale as in Culling: no need to pre-specify the number of scenes.

**Config:**
```yaml
narrative:
  enabled: true
  clustering_eps: 0.5         # DBSCAN epsilon (lower = tighter clusters)
```

---

### 9. Album Layout

**Module:** `src/layout/engine.py`
**What it does:** Automatically generates album page layouts from the processed photos.

Supported layout algorithms:
- **Fixed Partition:** Equal-sized grid
- **Fixed Columns:** Column-based masonry layout
- **Hero:** One large photo + smaller supporting shots
- **Dynamic Collage:** Varied aspect ratios
- **Mixed:** Randomly alternates between algorithms

---

### 10. Reporting

**Module:** `src/utils/pdf_gen.py` + inline JSON generation
**What it does:** Generates a summary of the pipeline run.

**Outputs:**
- `report.json` — Machine-readable full report with per-image scores and pipeline statistics
- `summary.pdf` — Client-ready PDF with overview metrics and image status table

**Library:** ReportLab for PDF generation — industry standard for programmatic PDF creation in Python.

---

## Configuration System

The pipeline is entirely driven by YAML configuration files. Each file defines which stages are active and their parameters.

**Schema:** `src/config/schema.py` (Pydantic models with validation and defaults)

### Running with a config:
```bash
cd wedding_ai_pipeline
$env:PYTHONPATH="."; python main.py --config configs/color_only_config.yaml
```

### Example: Color grading only (no culling, no restoration):
```yaml
culling:
  enabled: false
restoration:
  enabled: false
color_grading:
  enabled: true
  style: "cinematic"
  strength: 1.0
background_removal:
  enabled: false
```

### Example: Full pipeline with cutouts:
```yaml
culling:
  enabled: true
restoration:
  enabled: true
color_grading:
  enabled: true
  style: "film_kodak"
background_removal:
  enabled: true
  model: "birefnet-portrait"
```

---

## Filter Presets Reference

See [`color_theory.md`](color_theory.md) for the full specification of all 19 presets, including:
- Exact per-channel HSL adjustment values
- Tone curve parameters
- Split toning hues and saturations
- Vignette and grain settings
- Color psychology rationale for each preset
- SAM-based semantic segmentation strategy

---

## Technology Stack

| Category | Library | Version | Purpose |
|:---------|:--------|:--------|:--------|
| **Config** | Pydantic | 2.x | Schema validation, YAML parsing |
| **Image I/O** | OpenCV, Pillow | 4.x, 10.x | Read/write/convert images |
| **Culling** | MediaPipe | 0.10.x | Face mesh, blink detection |
| | ImageHash | 4.x | Perceptual hashing for dedup |
| | scikit-learn | 1.x | DBSCAN clustering |
| **Restoration** | PyTorch | 2.x | NAFNet / GFPGAN inference |
| **Color Grading** | OpenCV + NumPy | — | HSV/LAB, Generative 1D LUTs |
| **Background Removal** | rembg + BiRefNet | 2.x | Portrait segmentation |
| **Narrative** | OpenCLIP | — | CLIP embeddings for scene grouping |
| **Reporting** | ReportLab | 4.x | PDF generation |
| **Testing** | pytest | 9.x | Unit and integration tests |
| **Progress** | tqdm | 4.x | Progress bars |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd wedding_ai_pipeline

# Install dependencies
pip install -e .

# Download AI model weights (optional, for restoration)
python scripts/download_models.py
```

**Requirements:**
- Python 3.11+
- NVIDIA GPU with CUDA (recommended) or CPU
- Minimum 4GB VRAM (tiling enabled) or 8GB+ (tiling disabled)

---

## Usage

### Basic: Run with a configuration file
```bash
cd wedding_ai_pipeline
$env:PYTHONPATH="."
python main.py --config configs/color_only_config.yaml
```

### Dry run: Validate config without processing
```bash
python main.py --config configs/color_only_config.yaml --dry-run
```

### Output Structure
```
output/<pipeline_name>/
├── final/           ← Graded JPEGs
├── cutouts/         ← Background-removed PNGs (if enabled)
├── report.json      ← Machine-readable results
└── summary.pdf      ← Client-ready PDF report
```

---

## Project Structure

```
wedding_ai_pipeline/
├── configs/                    # YAML configuration files
│   ├── prod_config.yaml        # Full pipeline config
│   └── color_only_config.yaml  # Color grading + cutouts config
├── src/
│   ├── pipeline/
│   │   └── orchestrator.py     # Main pipeline orchestration
│   ├── config/
│   │   └── schema.py           # Pydantic config models
│   ├── culling/
│   │   ├── engine.py           # Culling orchestrator
│   │   ├── blur_detector.py    # Laplacian blur detection
│   │   ├── face_analyzer.py    # MediaPipe face analysis
│   │   ├── quality_assessor.py # BRISQUE/NIQE quality metrics
│   │   └── duplicate_cluster.py # Perceptual hash + DBSCAN
│   ├── restoration/
│   │   ├── engine.py           # Restoration router
│   │   ├── nafnet_restore.py   # NAFNet deblurring
│   │   └── gfpgan_restore.py   # Face restoration
│   ├── color/
│   │   └── engine.py           # AI Colorist (19 presets)
│   ├── segmentation/
│   │   └── background_remover.py # BiRefNet background removal
│   ├── cropping/
│   │   └── engine.py           # Smart cropping (RT-DETR)
│   ├── watermark/
│   │   └── engine.py           # Watermark overlay
│   ├── narrative/
│   │   └── engine.py           # CLIP + DBSCAN clustering
│   ├── layout/
│   │   └── engine.py           # Album layout generation
│   ├── utils/
│   │   ├── image_io.py         # Image load/save helpers
│   │   ├── tiling.py           # VRAM tiling for low-memory GPUs
│   │   ├── gpu.py              # Device detection (CUDA/CPU)
│   │   └── pdf_gen.py          # PDF report generation
│   └── core/
│       ├── models.py           # Data models (ImageScore, PipelineResult)
│       └── enums.py            # Enums (ColorStyle, CropRatio, etc.)
├── tests/                      # 18 test files
├── color_theory.md             # AI Colorist reference spec
├── main.py                     # CLI entry point
└── pyproject.toml              # Project metadata
```

---

## Testing

Run all tests:
```bash
$env:PYTHONPATH="."
python -m pytest tests/ -vv
```

Run a specific test file:
```bash
python -m pytest tests/test_color.py -vv
```

The test suite covers all major modules:
- `test_color.py` — All 19 color presets, strength control, B&W, pastel desaturation
- `test_background_removal.py` — RGBA output, dimension preservation, transparency
- `test_pipeline.py` — Pipeline orchestration, step management
- `test_blur.py` — Blur detection thresholds
- `test_config.py` — YAML parsing and Pydantic validation
- `test_tiling.py` — Image splitting and merging for VRAM management
- And 12 more covering every engine in the pipeline

---

## Git Checkpoints

| Tag | Description |
|:----|:------------|
| `v0.4-checkpoint` | Core pipeline + tiling + PDF report |
| `v0.5-colorist` | AI Colorist with 19 filter presets |

Revert to any checkpoint:
```bash
git checkout v0.5-colorist
```

---

## License

This project is designed for commercial use by wedding photographers. See `AGENTS.md` for licensing terms.
