---
description: TDD Red-Green-Refactor Lifecycle
---

Follow this workflow for every new feature or bug fix:

1. **🔴 Red**: 
   - Write a failing test in `tests/` that specifies the desired behavior.
   - Run `pytest` to verify the test fails.
   - // turbo
   - `python -m pytest tests/path_to_test.py`

2. **🟢 Green**: 
   - Write the minimum amount of code in `src/` to make the test pass.
   - Run `pytest` until it passes.
   - // turbo
   - `python -m pytest tests/path_to_test.py`

3. **🔵 Refactor**: 
   - Clean up code, remove duplication, and improve names.
   - Ensure tests stay green.
   - // turbo
   - `python -m pytest tests/path_to_test.py`

4. **✅ Checkpoint**:
   - Create a git commit.
   - DO NOT git push without EXPLICIT instructions
   - // turbo
   - `git add . && git commit -m "feat/test/refactor: brief description"`
