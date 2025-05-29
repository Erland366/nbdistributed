# GPU Assignment Examples

The jupyter_distributed extension now supports specifying specific GPU IDs for your distributed workers, which is particularly useful for mixed GPU topologies.

## Basic Usage

### Auto-assignment (Default behavior)
```python
%load_ext jupyter_distributed

# Workers will cycle through all available GPUs
%dist_init --num-ranks 4
```
This assigns GPUs automatically: rank 0 → GPU 0, rank 1 → GPU 1, rank 2 → GPU 2, rank 3 → GPU 3 (if 4 GPUs available)

### Specify GPU IDs
```python
%load_ext jupyter_distributed

# Use specific GPUs (e.g., only GPU 0 and GPU 1)
%dist_init --num-ranks 2 --gpu-ids 0,1
```

### Mixed topology example
```python
# For a system with different GPU types, use only the fast ones
%dist_init --num-ranks 2 --gpu-ids 0,3  # Skip GPUs 1,2
```

## Advanced Examples

### Check GPU assignments
```python
%%distributed
import torch
print(f"Rank {rank}: GPU ID = {gpu_id}")
print(f"Rank {rank}: Current device = {device}")
print(f"Rank {rank}: Device name = {torch.cuda.get_device_name()}")
```

### GPU cycling for many ranks
```python
# 4 ranks but only 2 GPUs specified - will cycle through them
%dist_init --num-ranks 4 --gpu-ids 0,1
# Result: rank 0→GPU 0, rank 1→GPU 1, rank 2→GPU 0, rank 3→GPU 1
```

### Validation and error handling
```python
# Invalid GPU ID
%dist_init --num-ranks 2 --gpu-ids 0,5  # Fails if GPU 5 doesn't exist

# Not enough GPU IDs  
%dist_init --num-ranks 4 --gpu-ids 0,1  # Fails - need 4 ranks, only 2 GPUs

# Invalid format
%dist_init --num-ranks 2 --gpu-ids abc,def  # Fails - not integers
```

## Complete Workflow Example

```python
# Load extension
%load_ext jupyter_distributed

# Start workers on specific GPUs
%dist_init --num-ranks 2 --gpu-ids 0,1

# Check status and GPU assignments
%dist_status

# Verify GPU assignment
%%distributed
print(f"Rank {rank} is using GPU {gpu_id}")
tensor = torch.randn(1000, 1000).cuda()
print(f"Tensor device: {tensor.device}")

# Run distributed training
%%distributed
import torch.nn as nn
model = nn.Linear(1000, 10).cuda()
print(f"Model on rank {rank} device: {next(model.parameters()).device}")

# Cleanup
%dist_shutdown
```

## Benefits

1. **Mixed topologies**: Use only specific GPUs (e.g., skip slower GPUs)
2. **Resource isolation**: Avoid conflicting with other processes using certain GPUs  
3. **Debugging**: Test on specific GPU configurations
4. **Memory management**: Control which GPUs are used based on memory availability

## Notes

- If CUDA is not available, GPU IDs are ignored and workers use CPU
- The extension validates GPU IDs against available devices
- Workers automatically set the correct CUDA device and use NCCL backend
- GPU assignments are available in worker namespace as `gpu_id` and `device` variables 