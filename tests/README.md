# Testing

## Test Structure

```
tests/
├── __init__.py           # Python path setup
├── conftest.py           # Shared fixtures
├── test_solver.py        # Tests for PuzzleSolver
└── test_multithreads.py  # Tests for MultiThreadPuzzleSolver
```

## Running Tests

### All tests with coverage
```bash
pytest
```

### Fast tests (skip slow ones)
```bash
pytest -m "not slow"
```

### Slow tests only
```bash
pytest -m "slow"
```

### Parallel execution (faster)
```bash
pytest -n auto
```

### Specific test
```bash
pytest tests/test_solver.py::TestPuzzleSolver::test_solve_with_two_pieces -v
```

### With print output
```bash
pytest -s
```

## Coverage Reports

### HTML report
After running tests, open:
```bash
htmlcov/index.html
```

### Terminal with details
```bash
pytest --cov=py --cov-report=term-missing
```

### JSON report
```bash
# Auto-generated in coverage.json
```

## Latest Results

✅ **11 tests passed** (10 fast, 1 slow)
📊 **Code coverage: 64.35%**
- `puzzle.py`: 96.32%
- `solver.py`: 96.30%
- `multithreadssolver.py`: 41.35%

### Fast tests: ~6 seconds
### All tests: ~2 minutes

## Fixtures

Use ready-made fixtures from `conftest.py`:

- `simple_pieces` - simple test pieces
- `small_board` - small board 6x7
- `medium_board` - medium board 7x7
- `calendar_pieces` - calendar puzzle pieces

Example:
```python
def test_my_feature(small_board, simple_pieces):
    solver = PuzzleSolver(small_board, [simple_pieces['A']])
    solutions, _, _ = solver.solve(findAll=True, printSol=False)
    assert len(solutions) > 0
```

## Markers

- `@pytest.mark.slow` - slow tests (>30 sec)
- `@pytest.mark.integration` - integration tests

## Configuration

- `pytest.ini` - pytest configuration
- `pyproject.toml` - coverage configuration
- `requirements-dev.txt` - development dependencies
