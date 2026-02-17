class PipelineError(Exception):
    """Base class for pipeline exceptions."""
    pass

class CorruptImageError(PipelineError):
    """Raised when an image file is corrupt."""
    pass

class ModelLoadError(PipelineError):
    """Raised when an AI model fails to load."""
    pass

class GPUMemoryError(PipelineError):
    """Raised when out of GPU memory."""
    pass

class ConfigValidationError(PipelineError):
    """Raised when config validation fails."""
    pass
