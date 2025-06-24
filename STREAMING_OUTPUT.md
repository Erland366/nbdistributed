# Streaming Output Feature for nbdistributed

## Problem Solved

Previously, when using `@/nbdistributed`, output would only appear **after** the entire cell finished executing. This meant:

- ❌ No progress bars during training loops
- ❌ No intermediate print statements
- ❌ No real-time feedback during long-running operations
- ❌ Difficult to monitor distributed training progress

## Solution: Real-Time Streaming Output

The new streaming output feature enables **real-time output display** similar to normal Jupyter notebook behavior.

### How It Works

1. **Worker Output Capture**: Each worker now uses a `StreamingOutput` class that immediately sends output to the coordinator
2. **Asynchronous Communication**: Output messages are processed immediately via callbacks in the communication manager
3. **Real-Time Display**: Output appears in the notebook as it's generated, not after completion

### Key Components

#### StreamingOutput Class (worker.py)
```python
class StreamingOutput:
    def write(self, text):
        if text and text.strip():
            # Send output immediately to coordinator
            stream_message = Message(
                msg_type="stream_output",
                rank=self.rank,
                data={"text": text, "stream": "stdout"}
            )
            self.socket.send(pickle.dumps(stream_message))
        
        # Also keep in buffer for final result
        self.buffer.write(text)
```

#### Communication Manager Updates (communication.py)
```python
def _message_handler(self):
    # Handle streaming output messages immediately
    if msg.msg_type == "stream_output":
        if self.output_callback:
            self.output_callback(msg.rank, data["text"], data["stream"])
        continue  # Don't queue streaming messages
```

#### Magic Class Updates (magic.py)
```python
def _handle_streaming_output(self, rank: int, text: str, stream_type: str):
    """Handle real-time streaming output from workers."""
    if stream_type == "result":
        print(f"[Rank {rank}] Out: {text}")
    else:
        print(f"[Rank {rank}] {text}", end="")
```

## Usage Examples

### Before (Buffered Output)
```python
%%distributed
for i in range(10):
    print(f"Step {i}")
    time.sleep(1)
print("Done!")

# Output appears only after 10 seconds:
# Step 0
# Step 1
# ...
# Step 9
# Done!
```

### After (Streaming Output)
```python
%%distributed
for i in range(10):
    print(f"Step {i}")
    time.sleep(1)
print("Done!")

# Output appears immediately as each step executes:
# 🚀 Executing on all ranks (streaming output enabled)...
# [Rank 0] Step 0
# [Rank 1] Step 0
# [Rank 0] Step 1
# [Rank 1] Step 1
# ... (continues in real-time)
```

### Progress Bars
```python
%%distributed
from tqdm import tqdm
import time

for i in tqdm(range(100), desc="Training"):
    # Simulate training step
    time.sleep(0.1)
    if i % 10 == 0:
        print(f"Checkpoint at step {i}")

# Progress bars now update in real-time!
```

### Training Loop Example
```python
%%distributed
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    for batch in range(100):
        # Training code here
        loss = train_step(batch)
        if batch % 20 == 0:
            print(f"  Batch {batch}, Loss: {loss:.4f}")
    print(f"Epoch {epoch+1} completed!")

# Real-time training progress visible throughout execution
```

## Benefits

✅ **Real-time feedback**: See progress as it happens  
✅ **Progress bars work**: tqdm and other progress libraries function correctly  
✅ **Better debugging**: Immediate output helps identify issues  
✅ **Jupyter-like experience**: Maintains familiar notebook behavior  
✅ **Rank identification**: Output clearly shows which worker generated it  
✅ **Backwards compatible**: Existing code works without changes  

## Technical Details

### Message Flow
1. Worker executes code line by line
2. Each `print()` or stdout write triggers `StreamingOutput.write()`
3. `StreamingOutput` immediately sends message to coordinator
4. Coordinator's `_message_handler` processes streaming messages instantly
5. `_handle_streaming_output` callback displays output in notebook
6. Final execution results are collected normally

### Performance Impact
- Minimal latency: Streaming messages are processed asynchronously
- No blocking: Code execution continues while output is streamed
- Efficient: Only actual output triggers network communication
- Scalable: Works with any number of worker processes

## Testing

Run the included test file to see streaming output in action:

```python
# In a notebook cell:
exec(open('test_streaming.py').read())
```

The test demonstrates:
- Basic streaming with print statements
- Progress bar functionality
- Training loop simulation
- Mixed output scenarios

## Comparison

| Feature | Before | After |
|---------|--------|-------|
| Output timing | After completion | Real-time |
| Progress bars | ❌ Not visible | ✅ Work perfectly |
| Long loops | ❌ No feedback | ✅ Live updates |
| Debugging | ❌ Difficult | ✅ Easy |
| Training monitoring | ❌ No visibility | ✅ Full visibility |

This streaming output feature transforms `@/nbdistributed` from a batch execution system into a truly interactive distributed computing environment that maintains the responsiveness users expect from Jupyter notebooks. 