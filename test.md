# Test Registry & Results

This file tracks the TDD progress, rationale for each test, and current status.

| Test Case | Rationale | Status | Comments |
|-----------|-----------|--------|----------|
| `test_config_loads_valid_yaml` | Ensures the system can correctly parse valid YAML configurations into Pydantic models. | ✅ PASSED | Core foundation for config-driven pipeline. |
| `test_config_rejects_invalid_yaml` | Ensures the system fails early with clear validation errors when given malformed config. | ✅ PASSED | Prevents runtime crashes due to bad config. |
| `test_image_score_initialization` | Validates the primary domain model for quality scoring. | ✅ PASSED | Essential for culling logic. |
| `test_load_save_jpeg` | Ensures basic image I/O works for standard formats. | ✅ PASSED | Basic requirement for processing. |
| `test_develop_raw_mock` | Validates the RAW development interface without needing a heavy RAW file fixture. | ✅ PASSED | Core for "RAW first" workflow. |
| `test_gpu_detection_returns_valid_string`| Ensures the system correctly identifies available hardware (CUDA/MPS/CPU). | ✅ PASSED | Critical for performance optimization on 1650/Apple Silicon. |
| `test_gpu_detection_respects_manual_override`| Validates that users can force a specific hardware backend. | ✅ PASSED | Useful for debugging/CI. |
| `test_pipeline_runs_and_returns_result` | Validates the orchestrator's basic execution flow. | ✅ PASSED | Foundation for multi-step processing. |
| `test_pipeline_add_step` | Ensures steps can be dynamically added to the pipeline. | ✅ PASSED | Essential for modular micro-tool architecture. |
| `test_blur_detector_low_variance_is_blurry`| Flags images with low Laplacian variance as blurry. | ✅ PASSED | Phase 2 culling logic. |
| `test_blur_detector_high_variance_is_sharp`| Correctly identifies sharp images. | ✅ PASSED | Phase 2 culling logic. |
| `test_duplicate_clusterer_groups_same_images`| Validates pHash clustering for identical files. | ✅ PASSED | Prevents album redundancy. |
| `test_duplicate_clusterer_separates_different_images`| Ensures distinct images are not incorrectly clustered. | ✅ PASSED | Prevents over-culling. |
| `test_quality_assessor_returns_scores`| Validates BRISQUE/NIQE scoring interface. | ✅ PASSED | Perceptual quality metrics. |
| `test_face_analyzer_detects_no_faces` | Verifies detection negative case. | ✅ PASSED | Transitioned to MediaPipe Tasks API. Foundation solid. |
| `test_face_analyzer_detects_blink_mock` | Validates EAR-based blink detection logic. | ✅ PASSED | Logic branch verified via mocking. |
| `test_culling_engine_scores_image` | Ensures the composite culling engine correctly aggregates scores from sub-engines. | ✅ PASSED | Brain of Stage 1/2 culling verified. |
| `test_restoration_engine_initialization`| Ensures the RestorationEngine can be initialized with default backends. | ✅ PASSED | Phase 3 restoration foundation. |
| `test_restoration_routing_logic` | Validates the auto-routing logic based on quality scores. | ✅ PASSED | Model selection logic verified. |
| `test_tiler_splits_and_merges` | Validates image splitting and merging for low-VRAM tiling. | ✅ PASSED | Critical for GTX 1650 support. |
| `test_color_grading_lab_statistical` | Ensures that statistical color transfer in Lab space works. | ✅ PASSED | Multi-shot color consistency foundation. |
| `test_narrative_engine_clustering` | Validates that the narrative engine can group images into chapters. | ✅ PASSED | Foundation of the wedding storyline. |
| `test_cropping_engine_square_crop` | Validates that the engine can produce a 1:1 square crop. | ✅ PASSED | Subject-aware cropping foundation. |
| `test_cropping_engine_story_crop` | Validates 9:16 story crop for social media. | ✅ PASSED | Vertical delivery format support. |
| `test_watermark_placement_bottom_right` | Verifies that the watermark can be placed in the bottom-right corner. | ✅ PASSED | Basic branding support. |
| `test_watermark_auto_placement` | Verifies the auto-placement logic based on image detail. | ✅ PASSED | Intelligent branding placement. |
| `test_layout_engine_fixed_partition` | Validates the fixed-partition layout algorithm. | ✅ PASSED | Automated album design foundation. |
| `test_pipeline_add_step` | Ensures steps can be dynamically added to the pipeline. | ✅ PASSED | Modular architecture verified. |
| `test_pipeline_runs_and_returns_result`| Validates the orchestrator's full execution flow from ingestion. | ✅ PASSED | End-to-end orchestration base verified. |
| `test_srgb_to_linear_black_is_zero` | Validates sRGB→Linear gamma decode at boundary (black). | ✅ PASSED | SOTA Engine P0 foundation. |
| `test_srgb_to_linear_white_is_one` | Validates sRGB→Linear gamma decode at boundary (white). | ✅ PASSED | SOTA Engine P0 foundation. |
| `test_srgb_to_linear_midgrey` | Validates gamma correction: sRGB 0.5 → linear ~0.214. | ✅ PASSED | Proves gamma transfer function is correct. |
| `test_linear_to_srgb_roundtrip` | sRGB→Linear→sRGB roundtrip must be identity within ε. | ✅ PASSED | Ensures no data loss in color space conversions. |
| `test_white_has_L_one` | White in Oklab should have L=1.0, a=0, b=0. | ✅ PASSED | Oklab matrix correctness. |
| `test_black_has_L_zero` | Black in Oklab should have L=0.0, a=0, b=0. | ✅ PASSED | Oklab matrix correctness. |
| `test_red_has_positive_a` | Red in Oklab should have positive 'a' axis. | ✅ PASSED | Validates Oklab perceptual geometry. |
| `test_oklab_roundtrip_accuracy` | sRGB→Oklab→sRGB roundtrip error must be <0.001. | ✅ PASSED | Critical for zero-degradation color grading. |
| `test_oklab_saturation` | Oklab chroma (perceptual saturation) computation. | ✅ PASSED | Foundation for subtractive saturation. |
| `test_oklab_hue` | Oklab hue angle computation via atan2. | ✅ PASSED | Foundation for per-channel color targeting. |
| `test_engine_accepts_float32_input` | SOTA engine V2 must accept float32 [0-1] input. | ✅ PASSED | P0: End-to-end float32 pipeline. |
| `test_engine_accepts_uint8_input` | SOTA engine V2 must accept uint8 for backward compat. | ✅ PASSED | Backward compatibility preserved. |
| `test_output_is_uint8` | Final output must be uint8 for file saving. | ✅ PASSED | File I/O compatibility. |
| `test_spline_curve_identity` | Linear spline (0,0)→(1,1) should be identity. | ✅ PASSED | P1: Cubic spline baseline. |
| `test_spline_curve_lifted_shadows` | Lifted black point spline has lut[0]>0. | ✅ PASSED | P1: Film fade simulation. |
| `test_spline_curve_rolled_highlights` | Rolled highlight spline has lut[-1]<1. | ✅ PASSED | P1: Highlight protection. |
| `test_spline_curve_s_shape` | S-curve spline increases contrast correctly. | ✅ PASSED | P1: Cinematic contrast curves. |
| `test_subtractive_darkens_saturated_colors` | Subtractive saturation makes saturated colors darker. | ✅ PASSED | P1: Filmic density emulation — the "film look". |
| `test_subtractive_preserves_neutral` | Subtractive saturation barely affects grey/neutral colors. | ✅ PASSED | P1: Neutral stability. |
| `test_all_presets_run_v2` | All 19 presets run through V2 without errors. | ✅ PASSED | Full backward compatibility smoke test. |
