# Installation Guide for jupyter_distributed

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher
- Jupyter Notebook or JupyterLab

## Installation

### 1. Install the Package

```bash
# Clone the repository
cd jupyter_distributed/

# Install in development mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

### 2. Verify Installation

```bash
# Run the test suite
python run_tests.py unit

# Or run a specific test
pytest tests/test_magic.py -v
```

### 3. Load the Extension in Jupyter

In any Jupyter notebook:

```python
%load_ext jupyter_distributed
```

## Quick Start

### Basic Usage

```python
# 1. Load extension
%load_ext jupyter_distributed

# 2. Initialize distributed workers
%dist_init --num-ranks 2

# 3. Execute code on all ranks
%%distributed
import torch
tensor = torch.randn(2, 3)
print(f"Rank {rank}: {tensor.shape}")

# 4. Execute on specific ranks
%%rank[0]
print("Only on rank 0")

# 5. Synchronize all ranks
%sync

# 6. Shutdown when done
%dist_shutdown
```

### Available Commands

| Command | Description |
|---------|-------------|
| `%dist_init --num-ranks N` | Start N distributed workers |
| `%dist_status` | Show status of all workers |
| `%dist_shutdown` | Shutdown all workers |
| `%%distributed` | Execute cell on all ranks |
| `%%rank[0,1,2]` | Execute cell on specific ranks |
| `%%rank[0-2]` | Execute cell on rank range |
| `%sync` | Synchronize all ranks (barrier) |

## Advanced Usage

### Complex Rank Specifications

```python
%%rank[0,2,4]       # Specific ranks
%%rank[0-3]         # Range of ranks  
%%rank[0,2-4,6]     # Mixed specification
```

### Error Handling

The extension gracefully handles errors and shows which rank failed:

```python
%%distributed
if rank == 1:
    raise ValueError("Error on rank 1")
else:
    print(f"Rank {rank} success")
```

### Collective Operations

```python
%%distributed
import torch.distributed as dist

# All-reduce
tensor = torch.randn(3, 3)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Broadcast
if rank == 0:
    data = torch.tensor([1, 2, 3])
else:
    data = torch.zeros(3)
dist.broadcast(data, src=0)
```

## Troubleshooting

### Common Issues

1. **Workers fail to start**
   - Check that PyTorch is installed
   - Ensure ports are available
   - Verify CUDA setup if using GPUs

2. **Communication timeouts**
   - Check firewall settings
   - Verify network connectivity
   - Increase timeout in code if needed

3. **Import errors**
   - Ensure package is installed: `pip list | grep jupyter-distributed`
   - Restart Jupyter kernel
   - Check Python path

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
%dist_init --num-ranks 2
```

### Testing

Run the full test suite:

```bash
# All tests
python run_tests.py

# Unit tests only (fast)
python run_tests.py unit

# Integration tests
python run_tests.py integration

# With coverage
python run_tests.py coverage
```

## Development

### Project Structure

```
jupyter_distributed/
├── src/jupyter_distributed/     # Main package
│   ├── __init__.py              # Extension loader
│   ├── magic.py                 # IPython magic commands
│   ├── communication.py         # ZeroMQ communication
│   ├── process_manager.py       # Worker process management
│   └── worker.py               # Distributed worker process
├── tests/                      # Comprehensive test suite
│   ├── test_*.py              # Individual test modules
│   ├── conftest.py            # Pytest configuration
│   └── requirements-test.txt   # Test dependencies
├── demo.ipynb                 # Usage demonstration
└── setup.py                  # Package configuration
```

### Contributing

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run tests before committing:
   ```bash
   python run_tests.py
   ```

3. Format code:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

### Architecture

- **Magic Commands**: IPython interface for user interaction
- **Process Manager**: Handles worker process lifecycle
- **Communication**: ZeroMQ-based message passing
- **Worker**: Distributed execution environment with PyTorch integration

The extension creates a coordinator process (main Jupyter kernel) that communicates with multiple worker processes via ZeroMQ. Each worker initializes PyTorch distributed and executes code in its own namespace.

## License

MIT License - see LICENSE file for details. 