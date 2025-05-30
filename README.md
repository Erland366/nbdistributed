# nbdistributed

A library for distributed PyTorch execution in Jupyter notebooks with seamless REPL-like behavior.

## Features

- **Seamless Distributed Execution**: Run PyTorch code across multiple GPUs directly from Jupyter notebooks
- **REPL-like Behavior**: See results immediately without explicit print statements
- **Automatic GPU Management**: Smart allocation of GPUs to worker processes
- **Interactive Development**: Real-time feedback and error reporting
- **IDE Support**: Namespace synchronization for code completion and type hints
- **Robust Process Management**: Graceful startup, monitoring, and shutdown

## Installation

```bash
pip install nbdistributed
```

## Quick Start

1. Import and initialize in your Jupyter notebook:

```python
%load_ext nbdistributed
%dist_init -n 4  # Start 4 worker processes
```

2. Run code on all workers:

```python
%%distributed
import torch
print(f"Rank {rank} running on {torch.cuda.get_device_name()}")
```

3. Run code on specific ranks:

```python
%%rank[0,1]
print(f"Running on rank {rank}")
```

## Architecture

The library consists of four main components:

### 1. Magic Commands (`magic.py`)
- Provides IPython magic commands for interaction
- Manages automatic distributed execution
- Handles namespace synchronization
- Key commands:
  - `%dist_init`: Initialize workers
  - `%%distributed`: Execute on all ranks
  - `%%rank[n]`: Execute on specific ranks
  - `%sync`: Synchronize workers
  - `%dist_status`: Show worker status
  - `%dist_mode`: Toggle automatic mode
  - `%dist_shutdown`: Clean shutdown

### 2. Worker Process (`worker.py`)
- Runs on each GPU/CPU
- Executes distributed PyTorch code
- Maintains isolated Python namespace
- Features:
  - REPL-like output capturing
  - Error handling and reporting
  - GPU device management
  - Namespace synchronization

### 3. Process Manager (`process_manager.py`)
- Manages worker lifecycle
- Handles GPU assignments
- Monitors process health
- Provides:
  - Clean process startup
  - Status monitoring
  - Graceful shutdown
  - GPU utilization tracking

### 4. Communication Manager (`communication.py`)
- Coordinates inter-process communication
- Uses ZMQ for efficient messaging
- Features:
  - Asynchronous message handling
  - Reliable message delivery
  - Timeout management
  - Worker targeting

## Usage Examples

### Basic Distributed Training

```python
%dist_init -n 2  # Start 2 workers

%%distributed
import torch
import torch.distributed as dist

# Create tensor on each GPU
x = torch.randn(100, 100).cuda()

# All-reduce across GPUs
dist.all_reduce(x)
print(f"Rank {rank}: {x.mean():.3f}")  # Same value on all ranks
```

### Selective Execution

```python
%%rank[0]
# Only runs on rank 0
model = torch.nn.Linear(10, 10).cuda()
print("Model created on rank 0")

%%distributed
# Broadcast model parameters to all ranks
for param in model.parameters():
    dist.broadcast(param.data, src=0)
print(f"Rank {rank} received model")
```

### GPU Information

```python
%dist_status
# Shows:
# - Process status
# - GPU assignments
# - Memory usage
# - Device names
```

### Automatic Mode

```python
%dist_mode --enable   # Enable automatic distributed execution
# Now all cells run on workers automatically

x = torch.randn(10, 10).cuda()
print(f"Rank {rank}: {x.mean()}")

%dist_mode --disable  # Disable automatic mode
# Now cells run locally unless explicitly distributed
```

## Advanced Features

### 1. GPU Assignment

Specify exact GPU-to-rank mapping:
```python
%dist_init -n 4 -g "0,1,2,3"  # Assign specific GPUs
```

### 2. Namespace Synchronization

The library automatically syncs worker namespaces to enable IDE features:
- Code completion
- Type hints
- Variable inspection

### 3. Error Handling

Errors are caught and reported with:
- Full traceback
- Rank information
- GPU context

### 4. Process Recovery

The library provides robust error recovery:
```python
%dist_reset    # Complete environment reset
%dist_init     # Start fresh
```

## Best Practices

1. **GPU Management**
   - Use `%dist_init` with `-g` for explicit GPU assignment
   - Monitor GPU usage with `%dist_status`
   - Clean up with `%dist_shutdown` when done

2. **Code Organization**
   - Use `%%distributed` for shared code
   - Use `%%rank[n]` for rank-specific operations
   - Keep model definitions in rank 0
   - Broadcast parameters to other ranks

3. **Error Handling**
   - Check `%dist_status` if workers seem unresponsive
   - Use `%dist_reset` for clean restart
   - Monitor GPU memory usage

4. **Performance**
   - Use appropriate batch sizes per GPU
   - Balance work across ranks
   - Synchronize only when necessary

## Troubleshooting

Common issues and solutions:

1. **Workers Won't Start**
   - Check GPU availability
   - Verify port availability
   - Look for Python environment issues

2. **Communication Errors**
   - Check network connectivity
   - Verify port accessibility
   - Ensure ZMQ installation

3. **GPU Issues**
   - Monitor memory usage
   - Check CUDA availability
   - Verify GPU assignments

4. **Process Hangs**
   - Use `%dist_status` to check state
   - Try `%dist_reset` for clean restart
   - Check for deadlocks

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
