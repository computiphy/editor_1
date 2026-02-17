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
