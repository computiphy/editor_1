from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Any, Dict
import cv2
from src.core.models import PipelineResult

class IStepEngine(Protocol):
    """Protocol all step engines must satisfy."""
    def run(self, input_dir: Path, output_dir: Path, config: Dict[str, Any]) -> List[Path]:
        ...

@dataclass
class PipelineStep:
    name: str
    enabled: bool
    engine: IStepEngine
    input_dir: Path
    output_dir: Path
    depends_on: List[str]

class WeddingPipeline:
    def __init__(self, config: Any):
        self.config = config
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def run(self) -> PipelineResult:
        """
        Execute the full pipeline: Ingestion -> Culling -> Restoration -> Grading -> Narrative -> Layout
        """
        import time
        from tqdm import tqdm
        from src.culling.engine import CullingEngine
        from src.restoration.engine import RestorationEngine
        from src.culling.blur_detector import BlurDetector
        from src.culling.face_analyzer import FaceAnalyzer
        from src.culling.quality_assessor import QualityAssessor
        from src.culling.duplicate_cluster import DuplicateClusterer
        
        start_time = time.time()
        
        # 1. Ingestion
        input_path = Path(self.config.pipeline.input_dir)
        photo_paths_set = set()
        for fmt in self.config.pipeline.input_formats:
            photo_paths_set.update(input_path.rglob(f"*.{fmt}"))
            photo_paths_set.update(input_path.rglob(f"*.{fmt.upper()}"))
        
        photo_paths = sorted(list(photo_paths_set))

        total_input = len(photo_paths)
        if total_input == 0:
            return PipelineResult(0, 0, 0, 0, 0, 0.0, [{"error": "No input photos found"}])

        # Initialize Engines
        culling_engine = None
        if self.config.culling.enabled:
            culling_engine = CullingEngine(
                blur_detector=BlurDetector(threshold=self.config.culling.blur_threshold),
                face_analyzer=FaceAnalyzer(),
                quality_assessor=QualityAssessor(),
                clusterer=DuplicateClusterer(threshold=self.config.culling.duplicate_threshold)
            )
            
        restoration_engine = None
        if self.config.restoration.enabled:
            restoration_engine = RestorationEngine(backend=self.config.pipeline.gpu_backend)
            
        grading_engine = None
        if self.config.color_grading.enabled:
            from src.color.engine import ColorGradingEngine
            grading_engine = ColorGradingEngine(
                method=self.config.color_grading.method,
                style=self.config.color_grading.style,
                strength=self.config.color_grading.strength,
                use_aces=self.config.color_grading.use_aces,
                perlin_grain=self.config.color_grading.perlin_grain,
                halation_enabled=self.config.color_grading.halation_enabled,
                clahe_enabled=self.config.color_grading.clahe_enabled
            )
            print(f"    Color Style: {self.config.color_grading.style} (strength={self.config.color_grading.strength})")

        # Standalone LUT engine (parsed once, reused for every image)
        lut_data = None
        if self.config.lut_application.enabled and self.config.lut_application.lut_path:
            from src.color.lut3d import parse_cube_file
            from pathlib import Path as _Path
            _lut_file = _Path(self.config.lut_application.lut_path)
            if _lut_file.exists():
                lut_data = parse_cube_file(str(_lut_file))
                print(f"    LUT Stage: {_lut_file.name} (intensity={self.config.lut_application.lut_intensity})")
            else:
                print(f"    Warning: LUT file not found: {_lut_file}")

        # ── Resolve stage-specific worker counts ───────────────────
        fallback = getattr(self.config.pipeline, 'workers', 4)
        workers_culling = getattr(self.config.culling, 'workers', None) or fallback
        workers_grading = getattr(self.config.color_grading, 'workers', None) or fallback
        workers_gpu     = getattr(self.config.background_removal, 'workers', None) or 1
        queue_maxsize   = getattr(self.config.pipeline, 'queue_maxsize', 4)

        print(f"    Workers: culling={workers_culling}, grading={workers_grading}, gpu={workers_gpu}, queue={queue_maxsize}")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue as queue_mod
        import threading

        # 2. Culling (parallelized)
        scores = []
        passed_images = []
        
        if self.config.culling.enabled:
            print(f"--- Stage 1: Culling {total_input} images (workers={workers_culling}) ---")
            def _cull_image(p):
                return culling_engine.evaluate_image(p)
            with ThreadPoolExecutor(max_workers=workers_culling) as cull_pool:
                futures = {cull_pool.submit(_cull_image, p): p for p in photo_paths}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        score = future.result()
                        scores.append(score)
                    except Exception as e:
                        print(f"Error evaluating {futures[future]}: {e}")
            passed_images = [s for s in scores if s.passed]
        else:
            print(f"--- Stage 1: Culling Disabled (Passing all {total_input} images) ---")
            from src.core.models import ImageScore
            for p in photo_paths:
                passed_images.append(ImageScore(
                    path=p,
                    blur_score=0.0,
                    fft_energy=0.0,
                    brisque_score=0.0,
                    niqe_score=0.0,
                    has_faces=False,
                    blink_detected=False,
                    expression_score=0.0,
                    overall_quality=1.0,
                    passed=True
                ))
                scores = passed_images

        # 3. Restoration & Grading & BG Removal
        total_restored = 0
        total_graded = 0
        
        output_dir = Path(self.config.pipeline.output_base) / self.config.pipeline.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)

        # LUT output lives in its own folder so you can diff easily
        lut_dir = None
        if self.config.lut_application.enabled and lut_data is not None:
            lut_dir = output_dir / "lut"
            lut_dir.mkdir(exist_ok=True)

        # Initialize Background Remover
        bg_remover = None
        if self.config.background_removal.enabled:
            from src.segmentation.background_remover import BackgroundRemover
            bg_remover = BackgroundRemover(model=self.config.background_removal.model, device=self.config.background_removal.device)

        # Determine active stage 2 features for dynamic labeling
        active_features = []
        if self.config.restoration.enabled: active_features.append("Restoration")
        if self.config.color_grading.enabled: active_features.append("Grading")
        if self.config.lut_application.enabled: active_features.append("LUT")
        
        stage_label = " + ".join(active_features) if active_features else "Processing"
        if not active_features and bg_remover:
            stage_label = "Caching"

        print(f"--- Stage 2: {stage_label} ({len(passed_images)} images) (workers={workers_grading}) ---")

        # Load reference image for grading if needed
        reference_img = None
        if self.config.color_grading.enabled and self.config.color_grading.reference_image:
             if Path(self.config.color_grading.reference_image).exists():
                 from src.utils.image_io import load_image
                 reference_img = load_image(self.config.color_grading.reference_image)
             else:
                 print(f"Warning: Reference image not found at {self.config.color_grading.reference_image}")

        # Create cutouts dir if bg removal is enabled
        cutouts_dir = None
        if self.config.background_removal.enabled:
            cutouts_dir = output_dir / "cutouts"
            cutouts_dir.mkdir(exist_ok=True)

        # Pre-initialize SemanticSegmenter to avoid race condition in threads
        if self.config.color_grading.enabled and self.config.color_grading.segmentation_enabled:
            from src.segmentation.semantic_segmenter import SemanticSegmenter
            self._segmenter = SemanticSegmenter()

        total_cutouts = 0

        from src.utils.image_io import load_image, save_image

        # ── Producer-Consumer Queue Architecture ─────────────────
        bg_queue = queue_mod.Queue(maxsize=queue_maxsize) if bg_remover else None

        def _grade_image(s):
            """Producer: load, restore, grade, save, then queue for BG removal."""
            counts = {"restored": 0, "graded": 0}
            try:
                img = load_image(str(s.path))
                
                # Restoration
                if self.config.restoration.enabled and restoration_engine:
                    img = restoration_engine.restore(
                        img, 
                        has_faces=s.has_faces, 
                        blur_score=s.blur_score
                    )
                    counts["restored"] = 1
                
                # Color Grading
                if self.config.color_grading.enabled and grading_engine:
                    img = grading_engine.apply_style(img)
                    
                    if reference_img is not None:
                        ref_result = grading_engine.apply_transfer(img, reference_img)
                        img = cv2.addWeighted(img, 0.7, ref_result, 0.3, 0)
                    
                    if self.config.color_grading.segmentation_enabled:
                        seg_result = self._segmenter.segment(img)
                        img = grading_engine.apply_semantic_grading(img, seg_result.as_dict())
                    
                    counts["graded"] = 1
                
                # Save graded image to final/
                rel_path = s.path.relative_to(input_path)
                target_save_path = final_dir / rel_path
                target_save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(img, str(target_save_path), output_format=self.config.pipeline.output_format)

                # LUT Application (stays in producer — needs graded img)
                if self.config.lut_application.enabled and lut_data is not None and lut_dir is not None:
                    from src.color.lut3d import apply_lut3d_array
                    import numpy as np
                    lut_input = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
                    lut_arr, lut_size = lut_data
                    lut_out = apply_lut3d_array(
                        lut_input, lut_arr, lut_size,
                        intensity=self.config.lut_application.lut_intensity
                    )
                    lut_img = np.clip(lut_out * 255.0, 0, 255).astype(np.uint8)
                    lut_save_path = lut_dir / rel_path
                    lut_save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(lut_img, str(lut_save_path), output_format=self.config.pipeline.output_format)

                # Queue graded image for BG removal (blocks if queue is full)
                if bg_queue is not None:
                    bg_queue.put((img, rel_path))

            except Exception as e:
                print(f"Error processing {s.path}: {e}")
                
            return counts

        def _bg_consumer(pbar):
            """Consumer: pull graded images from the queue, run BG removal, save cutouts."""
            local_cutouts = 0
            while True:
                item = bg_queue.get()
                if item is None:
                    bg_queue.task_done()
                    break  # Sentinel received — shut down
                img, rel_path = item
                try:
                    rgba = bg_remover.remove_background(img)
                    cutout_name = rel_path.with_suffix(".png")
                    target_cutout_path = cutouts_dir / cutout_name
                    target_cutout_path.parent.mkdir(parents=True, exist_ok=True)
                    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                    cv2.imwrite(str(target_cutout_path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    local_cutouts += 1
                except Exception as bg_err:
                    print(f"Error removing background for {rel_path}: {bg_err}")
                finally:
                    bg_queue.task_done()
                    pbar.update(1)
            return local_cutouts

        # ── Launch consumer threads first (they block on empty queue) ──
        consumer_futures = []
        consumer_pool = None
        bg_pbar = None
        if bg_remover:
            print(f"--- Stage 3: BG Removal (workers={workers_gpu}) ---")
            bg_pbar = tqdm(total=len(passed_images), desc="BG Removal")
            consumer_pool = ThreadPoolExecutor(max_workers=workers_gpu)
            for _ in range(workers_gpu):
                consumer_futures.append(consumer_pool.submit(_bg_consumer, bg_pbar))

        with ThreadPoolExecutor(max_workers=workers_grading) as grade_pool:
            futures = [grade_pool.submit(_grade_image, s) for s in passed_images]
            for future in tqdm(as_completed(futures), total=len(passed_images), desc=stage_label):
                res = future.result()
                total_restored += res["restored"]
                total_graded += res["graded"]

        # ── Sentinel shutdown: tell consumers to stop ─────────────
        if bg_queue is not None:
            for _ in range(workers_gpu):
                bg_queue.put(None)
            # Wait for all consumer threads to finish
            for cf in consumer_futures:
                total_cutouts += cf.result()
            consumer_pool.shutdown(wait=True)
            if bg_pbar:
                bg_pbar.close()
            print(f"--- Stage 3: BG Removal complete ({total_cutouts} cutouts) ---")
        # 4. Album Layout (Stage 9)
        total_album_pages = 0
        album_project = None
        if hasattr(self.config, 'layout') and self.config.layout.enabled:
            try:
                from src.layout.engine import AlbumLayoutEngine
                layout_engine = AlbumLayoutEngine(
                    mode=self.config.layout.mode,
                    page_size=tuple(self.config.layout.page_size),
                    dpi=self.config.layout.dpi,
                    images_per_page=self.config.layout.images_per_page,
                    padding=self.config.layout.padding,
                    gutter=self.config.layout.gutter,
                    use_cutouts=self.config.layout.use_cutouts,
                    background_dir=self.config.layout.background_directory,
                    background_strategy=self.config.layout.background_strategy,
                    background_seed=self.config.layout.background_seed,
                    export_format=self.config.layout.export.format,
                    export_quality=self.config.layout.export.quality,
                    ai_style=self.config.layout.ai_style,
                    ai_seed=self.config.layout.ai_seed,
                )
                album_project = layout_engine.generate_album(
                    final_dir=final_dir,
                    cutouts_dir=output_dir / "cutouts" if self.config.background_removal.enabled else None,
                    output_dir=output_dir,
                    config_snapshot=self.config.model_dump() if hasattr(self.config, 'model_dump') else {},
                )
                total_album_pages = len(album_project.pages)
            except Exception as e:
                print(f"Error in album layout: {e}")
                import traceback
                traceback.print_exc()

        # 5. Report
        import json
        from dataclasses import asdict
        
        report = {
            "summary": {
                "total_input": total_input,
                "total_passed": len(passed_images),
                "total_restored": total_restored,
                "total_graded": total_graded,
                "total_cutouts": total_cutouts,
                "total_album_pages": total_album_pages,
                "elapsed_seconds": time.time() - start_time
            },
            "images": [asdict(s) for s in scores]
        }
        with open(output_dir / "report.json", "w") as f:
            json.dump(report, f, indent=4, default=str)

        # 6. PDF Summary
        try:
            from src.utils.pdf_gen import PDFReportGenerator
            pdf_gen = PDFReportGenerator(output_dir / "summary.pdf")
            pdf_gen.generate(report)
            print(f"Summary report generated: {output_dir / 'summary.pdf'}")
        except Exception as e:
            print(f"Error generating PDF report: {e}")

        return PipelineResult(
            total_input=total_input,
            total_culled=total_input - len(passed_images),
            total_restored=total_restored,
            total_graded=total_graded,
            album_pages=total_album_pages,
            elapsed_seconds=time.time() - start_time,
            errors=[]
        )
