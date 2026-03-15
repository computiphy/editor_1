## [Unreleased]
### Added
- Created `changelog_trt-engine-build-speedup.md` to track branch changes.
- Implemented `ParallelWarmer` in `scripts/warm_trt_cache.py` using `ProcessPoolExecutor`.
- Added SOLID abstractions (`IModelRegistry`, `IEngineWarmer`, `DllDiscovery`) to the warmer script.
- Added file-based locking to `StandaloneTrtRegistry` to prevent race conditions.
- Added automated tests: `tests/test_trt_parallel_warmer.py`, `tests/test_registry_locking.py`, and `tests/test_cpu_load.py`.

### Changed
- Refactored `scripts/warm_trt_cache.py` for parallelization and SOLID compliance. (Commit: `b7d8ddd`)

