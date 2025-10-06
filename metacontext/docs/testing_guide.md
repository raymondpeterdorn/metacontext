# Testing Guide

## Overview

Metacontext has a comprehensive test suite with **90% coverage** across 41 tests, ensuring reliability and maintainability. The tests are organized by module for easy navigation and maintenance.

## Test Structure

## Test Structure

```
tests/
├── test_simple_api.py    # Main API and handler integration tests
├── test_llm_handler.py   # LLM integration tests (real API testing)
└── test_prompt_loader.py # Prompt system validation tests
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
make test

# Run tests with coverage report  
make test-cov

# Quick test (stops on first failure)
make q-test

# Run specific test files
poetry run pytest tests/test_simple_api.py -v
poetry run pytest tests/test_llm_handler.py -v
poetry run pytest tests/test_prompt_loader.py -v

# Run specific test
poetry run pytest tests/test_simple_api.py::test_basic_dataframe_contextualization -v
```

### Advanced Test Options

```bash
# Run tests with detailed output
poetry run pytest tests/ -v -s

# Run only failed tests from last run
poetry run pytest tests/ --lf

# Run tests in parallel (if pytest-xdist installed)
poetry run pytest tests/ -n auto

# Generate HTML coverage report
make test-cov
open htmlcov/index.html
```

## Coverage by Module

| Module | Coverage | Tests | Status |
|--------|----------|-------|---------|
| CLI | 95% | 11 | ✅ Excellent |
| Core | 100% | 5 | ✅ Complete |
| LLM Handler | 87% | 10 | ✅ Good |
| Parser | 88% | 8 | ✅ Good |  
| Prompt Loader | 89% | 5 | ✅ Good |
| **Overall** | **90%** | **41** | ✅ **Excellent** |

## Test Categories

### 1. CLI Tests (`test_cli.py`)
Tests the command-line interface functionality:
- Help and version commands
- Basic project analysis
- Configuration handling
- Error scenarios
- Output validation

### 2. Core Tests (`test_core.py`)
Tests the main MetacontextGenerator class:
- Initialization and configuration
- Context generation workflow
- File saving functionality
- End-to-end integration

### 3. LLM Handler Tests (`test_llm_handler.py`)
Tests LLM integration with proper mocking:
- OpenAI API integration
- Anthropic API integration
- Error handling and retries
- Response parsing
- Configuration validation

### 4. Parser Tests (`test_parser.py`)
Tests multi-language code parsing:
- File detection and classification
- Content extraction
- Dependency analysis
- Git integration
- Error handling

### 5. Prompt Loader Tests (`test_prompt_loader.py`)
Tests prompt template system:
- YAML template loading
- Variable substitution
- Error handling
- File system interactions

### 6. Integration Tests (`test_integration.py`)
Tests end-to-end workflows:
- Complete project analysis
- Real file system operations
- Component integration

## Test Best Practices

### Mocking Strategy
- **External APIs**: All LLM API calls are mocked to avoid external dependencies
- **File System**: Uses pytest's `tmp_path` fixture for isolated file operations
- **Git Operations**: Mocked to avoid requiring actual git repositories

### Test Isolation
- Each test is independent and can run in any order
- No shared state between tests
- Clean setup and teardown for each test

### Coverage Goals
- Maintain >90% overall coverage
- Each new feature must include comprehensive tests
- Edge cases and error conditions are tested

## Writing New Tests

### Test File Organization
```python
"""
Tests for the [module_name] module (metacontext.[module_name]).
"""

from unittest.mock import Mock, patch
import pytest

from metacontext.[module_name] import [ClassName]

def test_basic_functionality():
    """Test basic functionality with clear description."""
    # Arrange
    instance = ClassName()
    
    # Act
    result = instance.method()
    
    # Assert
    assert result is not None
```

### Naming Conventions
- Test files: `test_[module_name].py`
- Test functions: `test_[functionality]_[scenario]()`
- Use descriptive docstrings for each test

### Common Patterns
```python
# Testing with temporary files
def test_with_files(tmp_path):
    file_path = tmp_path / "test.py"
    file_path.write_text("content")
    # ... test logic

# Testing with mocks
@patch("module.external_dependency")
def test_with_mock(mock_dependency):
    mock_dependency.return_value = "expected"
    # ... test logic

# Testing error conditions
def test_error_handling():
    with pytest.raises(ValueError, match="expected error"):
        # ... code that should raise error
```

## Continuous Integration

### Local CI Checks
```bash
# Run all quality checks and tests
make ci

# Individual check commands
make lint        # Code linting
make format-check # Code formatting
make type-check  # Type checking
make test        # Run tests
```

### Pre-commit Workflow
```bash
# Before committing changes
make pre-commit  # Fix issues and run tests
```

### GitHub Actions
The project is ready for GitHub Actions CI/CD with the following workflow:
```yaml
# .github/workflows/ci.yml (planned)
- Lint with Ruff
- Type check with MyPy  
- Test with pytest
- Coverage reporting
- Multi-Python version testing
```

## Test Maintenance

### Adding Tests for New Features
1. Create test file following naming convention
2. Write comprehensive tests covering:
   - Happy path scenarios
   - Edge cases
   - Error conditions
   - Integration points
3. Ensure coverage stays >90%
4. Update this documentation

### Debugging Test Failures
```bash
# Run specific failing test with debugging
poetry run pytest tests/test_module.py::test_function -v -s --pdb

# Show test output (remove -q flag)
poetry run pytest tests/ -v -s

# Run with coverage to see what's missing
poetry run pytest tests/ --cov=src/metacontext --cov-report term-missing
```

## Performance Testing

### Current Test Performance
- **Total runtime**: ~1 second for all 41 tests
- **Individual test average**: ~25ms per test
- **Coverage calculation**: ~200ms additional

### Optimization Guidelines
- Use mocks instead of real external calls
- Minimize file system operations
- Use `tmp_path` for temporary files
- Keep test data small and focused

## Documentation Testing

Tests also serve as documentation:
- Each test demonstrates expected usage
- Test names describe functionality
- Test assertions show expected behavior
- Mock configurations show integration patterns

This comprehensive test suite ensures the reliability and maintainability of Metacontext while providing excellent documentation of expected behavior.
