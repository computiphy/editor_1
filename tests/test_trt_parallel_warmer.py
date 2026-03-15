import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import concurrent.futures

# We want to test the ParallelWarmer class (which doesn't exist yet)
# and its ability to distribute work to multiple processes.

def test_parallel_warmer_distributes_work():
    """
    TDD Red Phase: Verify that ParallelWarmer correctly utilizes ProcessPoolExecutor.
    This test will fail because ParallelWarmer is not yet implemented in scripts/warm_trt_cache.py.
    """
    from scripts.warm_trt_cache import ParallelWarmer, IEngineWarmer, IModelRegistry
    
    # Mocks for dependencies
    mock_registry = MagicMock(spec=IModelRegistry)
    mock_registry.get_registry.return_value = {}
    mock_registry.cache_dir = Path(".trt_engine_cache")
    
    mock_warmer = MagicMock(spec=IEngineWarmer)
    mock_warmer.warm.return_value = (True, "mock_engine.engine")
    
    models = ["model1", "model2", "model3"]
    
    # We expect ParallelWarmer to take a list of models and a warmer implementation
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor, \
         patch("concurrent.futures.as_completed") as mock_as_completed:
        
        # Configure the executor mock
        executor_instance = mock_executor.return_value.__enter__.return_value
        
        # Mock futures
        mock_futures = []
        for m in models:
            f = MagicMock(spec=concurrent.futures.Future)
            f.result.return_value = (True, f"engine_{m}.engine")
            mock_futures.append(f)
            
        executor_instance.submit.side_effect = mock_futures
        mock_as_completed.return_value = mock_futures
        
        warmer_orchestrator = ParallelWarmer(
            models=models,
            base_warmer=mock_warmer,
            registry=mock_registry,
            max_workers=2
        )
        
        # Force cache check to return empty so it doesn't skip
        mock_registry.get_registry.return_value = {}
        
        results = warmer_orchestrator.run()
        
        # Verify executor was created with max_workers=2
        mock_executor.assert_called_once_with(max_workers=2)
        
        # Verify submit was called for each model
        assert executor_instance.submit.call_count == len(models)
        
        # Verify as_completed was called
        mock_as_completed.assert_called_once()
        
        # Check results
        assert len(results) == len(models)
        for m in models:
            assert results[m][0] is True
            assert results[m][1] == f"engine_{m}.engine"


if __name__ == "__main__":
    pytest.main([__file__])
