# Task List - Wedding AI Pipeline

## Phase 1: Foundation (Weeks 1–3)
- [x] Project scaffolding, `pyproject.toml`, Docker base image (Scaffolding done, Docker deferred)
- [x] Core domain models (`models.py`, `enums.py`, `exceptions.py`)
- [x] Config schema + YAML loader with Pydantic validation
- [x] Utility modules (image I/O, logging, GPU detection, parallelism)
- [x] RAW engine (rawpy backend)
- [x] Pipeline orchestrator full execution logic
- [x] CLI entry point (`main.py`)
- [x] Unit tests for core + config + raw engine

## Phase 2: Culling & Quality (Weeks 4–5)
- [x] Blur detection (Python implementation)
- [ ] Blur detection (C++ pybind11 accelerated version)
- [x] Perceptual hashing + duplicate clustering
- [x] BRISQUE/NIQE quality assessment (Baseline implementation)
- [x] Face analysis (MediaPipe Landmarks, EAR logic)
- [x] CullingEngine composite scoring + filtering
- [ ] Integration tests with sample images
- [ ] Graceful culling without EXIF metadata fallback

## Phase 3: AI Processing (Weeks 6–8)
- [/] NAFNet + GFPGAN restoration + model download script (Tiling implemented, Torch integrated)
- [x] VRAM tiling strategy (4 GB safe)
- [ ] Restoration self-correction
- [x] Color grading engine (Reinhard Lab transfer + LUT)
- [ ] GPU memory management + Apple Silicon MPS support

## Phase 3.5: SOTA Color Engine (Weeks 8–10)
- [x] P0: Oklab color space module (`src/color/oklab.py`) — pure NumPy sRGB↔Oklab transforms
- [x] P0: Eliminate uint8 roundtrips — 32-bit float throughout the grading pipeline
- [x] P1: Cubic spline tone curves via `scipy.interpolate.CubicSpline`
- [x] P1: Subtractive saturation (filmic density emulation in Oklab)
- [x] P1: Chromatic adaptation white balance (CAT02 via `colour-science`)
- [x] P2: Guided filter mask refinement (`cv2.ximgproc.guidedFilter`)
- [x] P2: Frequency separation for skin grading
- [x] P2: CLAHE in Oklab for flat/muddy images
- [x] P3: Halation (red-channel scattering) optical effect
- [x] P3: Luminance-mapped film grain (Perlin-style, not Gaussian)
- [x] P4: 3D LUT engine (.cube) with tetrahedral interpolation
- [x] P5: ACES/OCIO color management pipeline (via OpenColorIO)
- [ ] Re-tune all 19 presets for the new Oklab pipeline
- [ ] Update README, test.md, and color_theory.md

## Phase 4: Creative Output (Weeks 9–10)
- [ ] Segmentation module
- [x] Smart cropping (RT-DETR centering logic)
- [x] Watermark engine
- [x] Narrative engine with CLIP fallback
- [x] Layout engine

## Phase 5: Polish & Production (Weeks 11–12)
- [ ] End-to-end integration testing
- [ ] Docker GPU + CPU images + docker-compose
- [ ] Performance benchmarking
- [ ] Documentation
