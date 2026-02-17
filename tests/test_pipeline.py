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
