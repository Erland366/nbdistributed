# Test Suite for jupyter_distributed

This directory contains comprehensive tests for the jupyter_distributed package.

## Test Structure

```
tests/
├── __init__.py                 # Test package init
├── conftest.py                # Pytest fixtures and configuration
├── pytest.ini                # Pytest settings
├── requirements-test.txt      # Test dependencies
├── test_communication.py      # Communication layer tests
├── test_process_manager.py    # Process management tests
├── test_worker.py            # Worker process tests
├── test_magic.py             # Magic command tests
└── test_integration.py       # Integration and notebook tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
# From project root
pytest

# Or from tests directory
cd tests && pytest
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest -m "not slow and not integration"

# Integration tests
pytest -m integration

# Slow tests (may take longer)
pytest -m slow

# Specific test file
pytest tests/test_communication.py

# Specific test
pytest tests/test_magic.py::TestDistributedMagic::test_dist_init_success
```

### Generate Coverage Report

```bash
pytest --cov=jupyter_distributed --cov-report=html
```

## Test Categories

### Unit Tests
- **test_communication.py**: Tests ZeroMQ communication layer
- **test_process_manager.py**: Tests worker process lifecycle management
- **test_worker.py**: Tests distributed worker functionality
- **test_magic.py**: Tests IPython magic commands

### Integration Tests  
- **test_integration.py**: End-to-end system tests, notebook execution tests

## Test Features

### Mock Infrastructure
Tests use extensive mocking to avoid:
- Actual PyTorch distributed initialization (requires multiple processes)
- Real network communication (uses mock ZMQ sockets)
- Subprocess creation (uses mock Popen)

### Custom Notebook Testing
The `NotebookTestRunner` class allows testing Jupyter notebooks programmatically:

```python
with NotebookTestRunner(num_ranks=2) as runner:
    cells = [
        "%load_ext jupyter_distributed",
        "%dist_init --num-ranks 2",
        "%%distributed\nprint('Hello')"
    ]
    notebook_path = runner.create_test_notebook(cells)
    nb, error = runner.execute_notebook(notebook_path)
```

### Fixtures
Key pytest fixtures in `conftest.py`:
- `temp_dir`: Temporary directory for tests
- `free_port`: Available network port
- `comm_manager`: Communication manager instance
- `process_manager`: Process manager instance
- `mock_jupyter_kernel`: Mocked IPython environment

## Test Markers

- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.integration`: End-to-end integration tests
- `@pytest.mark.unit`: Fast unit tests

## Writing New Tests

### Adding Unit Tests
```python
class TestNewFeature:
    def test_feature_functionality(self):
        # Arrange
        setup_test_data()
        
        # Act
        result = call_feature()
        
        # Assert
        assert result == expected_value
```

### Adding Integration Tests
```python
@pytest.mark.integration
class TestNewIntegration:
    def setup_method(self):
        DistributedMagic.shutdown_all()
    
    def teardown_method(self):
        DistributedMagic.shutdown_all()
    
    def test_feature_integration(self):
        # Test complete workflow
        pass
```

### Mocking Guidelines
- Mock external dependencies (subprocess, torch.distributed, zmq)
- Use `patch` for temporary mocking
- Use fixtures for reusable mocks
- Mock at the boundary of your code

## Debugging Failed Tests

### Verbose Output
```bash
pytest -v -s  # -s shows print statements
```

### Single Test with Full Traceback
```bash
pytest tests/test_magic.py::TestDistributedMagic::test_dist_init_success -vvv --tb=long
```

### With PDB Debugger
```bash
pytest --pdb tests/test_magic.py::TestDistributedMagic::test_dist_init_success
```

## Continuous Integration

These tests are designed to run in CI environments without requiring:
- Multiple GPUs
- Actual distributed PyTorch setup
- Real network communication

All external dependencies are mocked for reliability and speed. 