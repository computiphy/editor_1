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
            photo_paths_set.update(input_path.glob(f"*.{fmt}"))
            photo_paths_set.update(input_path.glob(f"*.{fmt.upper()}"))
        
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
                style=self.config.color_grading.style,
                strength=self.config.color_grading.strength
            )
            print(f"    Color Style: {self.config.color_grading.style} (strength={self.config.color_grading.strength})")

        # 2. Sequential Execution
        scores = []
        passed_images = []
        
        if self.config.culling.enabled:
            print(f"--- Stage 1: Culling {total_input} images ---")
            for p in tqdm(photo_paths):
                try:
                    score = culling_engine.evaluate_image(p)
                    scores.append(score)
                except Exception as e:
                    print(f"Error evaluating {p}: {e}")
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

        # 3. Restoration & Grading & Saving
        total_restored = 0
        total_graded = 0
        
        output_dir = Path(self.config.pipeline.output_base) / self.config.pipeline.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)

        print(f"--- Stage 2: Processing {len(passed_images)} images ---")
        
        # Initialize Background Remover
        bg_remover = None
        if self.config.background_removal.enabled:
            from src.segmentation.background_remover import BackgroundRemover
            bg_remover = BackgroundRemover(model=self.config.background_removal.model)

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

        total_cutouts = 0

        for s in tqdm(passed_images):
            try:
                from src.utils.image_io import load_image, save_image
                from PIL import Image as PILImage
                img = load_image(str(s.path))
                
                # Restoration
                if self.config.restoration.enabled and restoration_engine:
                    img = restoration_engine.restore(
                        img, 
                        has_faces=s.has_faces, 
                        blur_score=s.blur_score
                    )
                    total_restored += 1
                
                # Color Grading
                if self.config.color_grading.enabled and grading_engine:
                    # Step 1: Apply style preset (cinematic, pastel, etc.)
                    img = grading_engine.apply_style(img)
                    
                    # Step 2: Optional reference transfer blend
                    if reference_img is not None:
                        ref_result = grading_engine.apply_transfer(img, reference_img)
                        # Blend at 30% to add reference vibe without overpowering preset
                        img = cv2.addWeighted(img, 0.7, ref_result, 0.3, 0)
                    
                    # Step 3: Semantic per-region overrides (skin, sky, vegetation, etc.)
                    if self.config.color_grading.segmentation_enabled:
                        from src.segmentation.semantic_segmenter import SemanticSegmenter
                        if not hasattr(self, '_segmenter'):
                            self._segmenter = SemanticSegmenter()
                        seg_result = self._segmenter.segment(img)
                        img = grading_engine.apply_semantic_grading(img, seg_result.as_dict())
                    
                    total_graded += 1
                
                # Save graded image to final/
                save_image(img, str(final_dir / s.path.name))

                # Background Removal (on the already-graded image)
                if self.config.background_removal.enabled and bg_remover:
                    try:
                        rgba = bg_remover.remove_background(img)
                        # Save as PNG to preserve transparency
                        cutout_name = Path(s.path).stem + ".png"
                        pil_rgba = PILImage.fromarray(rgba)
                        pil_rgba.save(str(cutouts_dir / cutout_name), "PNG")
                        total_cutouts += 1
                    except Exception as bg_err:
                        print(f"Error removing background for {s.path}: {bg_err}")
                
            except Exception as e:
                print(f"Error processing {s.path}: {e}")

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
                    export_format=self.config.layout.export.format,
                    export_quality=self.config.layout.export.quality,
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
