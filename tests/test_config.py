import pytest
from pathlib import Path
from src.config.loader import load_config

def test_config_loads_valid_yaml():
    """
    Rationale: Ensures the system can correctly parse valid YAML configurations into Pydantic models.
    Details: This test is the core foundation for the config-driven pipeline. It verifies that the `load_config` 
    function can handle standard fields like pipeline name, gpu_backend, and input_formats.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "valid_config.yaml"
    config = load_config(str(fixture_path))
    
    assert config.pipeline.name == "test_shoot"
    assert config.pipeline.gpu_backend == "cpu"
    assert "raw" in config.pipeline.input_formats

def test_config_rejects_invalid_yaml():
    """
    Rationale: Ensures the system fails early with clear validation errors when given malformed config.
    Details: This test prevents runtime crashes due to bad config by verifying that Pydantic 
    correctly identifies missing or malformed fields (e.g., missing input_dir).
    """
    fixture_path = Path(__file__).parent / "fixtures" / "invalid_config.yaml"
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        load_config(str(fixture_path))
