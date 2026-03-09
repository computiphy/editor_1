import pytest
from pathlib import Path
from src.pipeline.orchestrator import WeddingPipeline, PipelineStep, IStepEngine
from src.core.models import PipelineResult

class MockEngine(IStepEngine):
    def run(self, input_dir: Path, output_dir: Path, config: dict) -> list[Path]:
        return [output_dir / "test.jpg"]

def test_pipeline_runs_and_returns_result(tmp_path):
    """
    Rationale: Validates the orchestrator's basic execution flow.
    Details: This test ensures that the `WeddingPipeline` can initialize and execute to completion, 
    returning a valid `PipelineResult` object.
    """
    from unittest.mock import MagicMock
    mock_config = MagicMock()
    mock_config.pipeline.input_dir = str(tmp_path)
    mock_config.pipeline.input_formats = ["jpg"]
    
    pipeline = WeddingPipeline(config=mock_config)
    result = pipeline.run()
    assert isinstance(result, PipelineResult)
    assert result.elapsed_seconds >= 0

def test_pipeline_add_step():
    """
    Rationale: Ensures steps can be dynamically added to the pipeline.
    Details: This test validates the modular design of the pipeline, ensuring that different 
    engines (culling, restoration, etc.) can be plugged into the main execution flow.
    """
    pipeline = WeddingPipeline(config={})
    engine = MockEngine()
    step = PipelineStep(
        name="test",
        enabled=True,
        engine=engine,
        input_dir=Path("/tmp/in"),
        output_dir=Path("/tmp/out"),
        depends_on=[]
    )
    pipeline.add_step(step)
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].name == "test"

def test_pipeline_stage_specific_workers(tmp_path):
    """
    Rationale: Ensures the pipeline uses stage-specific worker counts.
    Details: Tests that workers_culling, workers_grading, workers_gpu, and
    queue_maxsize config fields are accepted and the pipeline completes.
    """
    from unittest.mock import MagicMock
    import cv2
    import numpy as np

    # Create dummy images
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    for i in range(4):
        cv2.imwrite(str(in_dir / f"test_{i}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

    mock_config = MagicMock()
    mock_config.pipeline.input_dir = str(in_dir)
    mock_config.pipeline.input_formats = ["jpg"]
    mock_config.pipeline.output_base = str(tmp_path / "output")
    mock_config.pipeline.name = "test_run"
    mock_config.pipeline.output_format = "jpeg"
    # Stage-specific worker counts
    mock_config.pipeline.workers = 2            # Legacy fallback
    mock_config.pipeline.workers_culling = 4
    mock_config.pipeline.workers_grading = 3
    mock_config.pipeline.workers_gpu = 1
    mock_config.pipeline.queue_maxsize = 2
    mock_config.culling.enabled = False
    mock_config.restoration.enabled = False
    mock_config.color_grading.enabled = False
    mock_config.lut_application.enabled = False
    mock_config.watermark.enabled = False
    mock_config.background_removal.enabled = False
    mock_config.narrative.enabled = False
    mock_config.layout.enabled = False
    mock_config.cropping.enabled = False
    
    pipeline = WeddingPipeline(config=mock_config)
    result = pipeline.run()
    assert result.total_input == 4
    assert result.elapsed_seconds > 0

def test_pipeline_producer_consumer_queue(tmp_path):
    """
    Rationale: Validates the Producer-Consumer queue architecture for BG removal.
    Details: Grading threads produce graded images; BG removal threads consume from
    a bounded queue. Tests that sentinel shutdown works and no deadlocks occur.
    """
    from unittest.mock import MagicMock, patch
    import cv2
    import numpy as np

    # Create dummy images
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    for i in range(3):
        img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        cv2.imwrite(str(in_dir / f"test_{i}.jpg"), img)

    mock_config = MagicMock()
    mock_config.pipeline.input_dir = str(in_dir)
    mock_config.pipeline.input_formats = ["jpg"]
    mock_config.pipeline.output_base = str(tmp_path / "output")
    mock_config.pipeline.name = "test_queue"
    mock_config.pipeline.output_format = "jpeg"
    mock_config.pipeline.workers = 1
    mock_config.pipeline.workers_culling = 2
    mock_config.pipeline.workers_grading = 2
    mock_config.pipeline.workers_gpu = 1
    mock_config.pipeline.queue_maxsize = 2
    mock_config.culling.enabled = False
    mock_config.restoration.enabled = False
    mock_config.color_grading.enabled = False
    mock_config.lut_application.enabled = False
    mock_config.watermark.enabled = False
    mock_config.background_removal.enabled = True
    mock_config.background_removal.model = "u2net"
    mock_config.background_removal.device = "cpu"
    mock_config.narrative.enabled = False
    mock_config.layout.enabled = False
    mock_config.cropping.enabled = False

    # Mock rembg to avoid downloading models in tests
    fake_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    with patch("src.segmentation.background_remover.BackgroundRemover.remove_background", return_value=fake_rgba):
        pipeline = WeddingPipeline(config=mock_config)
        result = pipeline.run()
    
    assert result.total_input == 3
    assert result.elapsed_seconds > 0
    # Verify cutouts directory was created and pipeline didn't deadlock
    cutouts_dir = tmp_path / "output" / "test_queue" / "cutouts"
    assert cutouts_dir.exists()

