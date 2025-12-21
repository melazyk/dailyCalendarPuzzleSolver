# Make Commands Cheat Sheet

## Basic Commands

| Command | Description | Execution Time |
|---------|-------------|----------------|
| `make help` | Show all available commands | instant |
| `make test` | Run all tests with coverage | ~2 minutes |
| `make test-fast` | Fast tests only | ~6 seconds |
| `make test-slow` | Slow tests only | ~2 minutes |
| `make clean` | Clean temporary files | instant |

## Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Install development dependencies |
| `make test-cov` | Tests with detailed coverage report |
| `make test-html` | Tests + open HTML coverage report |
| `make test-parallel` | Run tests in parallel (faster) |
| `make coverage-report` | Show coverage report from last run |
| `make coverage-json` | Export coverage to JSON |

## Usage Examples

### Daily Development
```bash
# Before starting work
make test-fast

# After making changes
make test-fast

# Before committing
make test
```

### Coverage Check
```bash
# Quick check
make test-cov

# Detailed analysis
make test-html  # opens browser with report
```

### Cleanup
```bash
# Clean all temporary files
make clean

# Reinstall dependencies
make clean install
```

## Environment Variables

By default, `python3` is used. To use a different Python:

```bash
make test PYTHON=/path/to/python
```

## Tips

1. **Use `make test-fast`** for quick checks during development
2. **Run `make test`** before committing to verify all tests
3. **Use `make clean`** if something behaves strangely
4. **Check `make test-html`** for code coverage analysis
