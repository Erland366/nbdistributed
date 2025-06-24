"""
Worker process implementation for distributed execution in Jupyter notebooks.

This module implements the worker process that runs on each GPU/CPU for distributed
execution. Each worker:
- Initializes PyTorch distributed environment
- Sets up ZMQ communication with the coordinator
- Maintains its own Python namespace
- Executes code sent from the notebook
- Handles REPL-like behavior with streaming output support

The workers are managed by the ProcessManager and communicate through the
CommunicationManager using ZMQ sockets.
"""

import os
import sys
import zmq
import pickle
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional
import traceback
import time
import threading
from io import StringIO
from nbdistributed.communication import Message


class StreamingOutput:
    """
    Custom output handler that streams output to coordinator as it's generated.
    
    This class intercepts stdout/stderr writes and immediately sends them to the
    coordinator, enabling real-time output display similar to normal Jupyter behavior.
    """
    
    def __init__(self, socket, rank, msg_id):
        self.socket = socket
        self.rank = rank
        self.msg_id = msg_id
        self.buffer = StringIO()
        self._lock = threading.Lock()
        
    def write(self, text):
        if text and text.strip():
            # Send output immediately to coordinator
            with self._lock:
                try:
                    # Send streaming output message
                    stream_message = Message(
                        msg_id=f"{self.msg_id}_stream_{time.time()}",
                        msg_type="stream_output",
                        rank=self.rank,
                        data={"text": text, "stream": "stdout"},
                        timestamp=time.time()
                    )
                    self.socket.send(pickle.dumps(stream_message))
                except Exception:
                    pass  # Don't break execution if streaming fails
        
        # Also keep in buffer for final result
        self.buffer.write(text)
        
    def flush(self):
        pass  # No-op for compatibility
        
    def getvalue(self):
        return self.buffer.getvalue()


class DistributedWorker:
    """
    Worker process for distributed execution of PyTorch code.
    
    This class represents a single worker in the distributed environment. Each worker:
    - Runs on a specific GPU (if available)
    - Participates in PyTorch distributed training
    - Maintains its own Python namespace
    - Executes code sent from the Jupyter notebook
    - Captures and returns output in a REPL-like manner with streaming support
    
    Attributes:
        rank (int): Global rank of this worker
        world_size (int): Total number of workers
        master_addr (str): Address of the master node
        master_port (str): Port for PyTorch distributed
        gpu_id (Optional[int]): Specific GPU ID assigned to this worker
        namespace (dict): Local Python namespace for code execution
        context (zmq.Context): ZMQ context for communication
        socket (zmq.Socket): ZMQ socket for coordinator communication
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        comm_port: int,
        gpu_id: Optional[int] = None,
    ):
        """
        Initialize a distributed worker process.
        
        Args:
            rank (int): Global rank of this worker
            world_size (int): Total number of workers
            master_addr (str): Address of the master node
            master_port (str): Port for PyTorch distributed
            comm_port (int): Port for ZMQ communication
            gpu_id (Optional[int]): Specific GPU ID to use, if any
            
        The initialization process:
        1. Sets up PyTorch distributed environment variables
        2. Initializes the distributed process group
        3. Sets up ZMQ communication with coordinator
        4. Initializes the local namespace with common variables
        5. Configures GPU if available
        """
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.gpu_id = gpu_id

        # Set up environment for PyTorch distributed
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        # Initialize PyTorch distributed
        if torch.cuda.is_available():
            if gpu_id is not None:
                # Use specified GPU ID
                torch.cuda.set_device(gpu_id)
                print(f"Worker {rank} using GPU {gpu_id}")
            else:
                # Fall back to cycling through available GPUs
                device_id = rank % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                print(f"Worker {rank} using GPU {device_id} (auto-assigned)")
            backend = "nccl"
        else:
            backend = "gloo"
            if gpu_id is not None:
                print(f"Worker {rank}: CUDA not available, ignoring GPU ID {gpu_id}")

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        # Set up communication with coordinator
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, f"worker_{rank}".encode())
        self.socket.connect(f"tcp://{master_addr}:{comm_port}")

        # Local namespace for code execution
        self.namespace = {
            "torch": torch,
            "dist": dist,
            "rank": rank,
            "world_size": world_size,
            "__rank__": rank,
            "__world_size__": world_size,
        }

        # Add GPU info to namespace
        if torch.cuda.is_available():
            self.namespace["gpu_id"] = (
                gpu_id if gpu_id is not None else rank % torch.cuda.device_count()
            )
            self.namespace["device"] = torch.cuda.current_device()
        else:
            self.namespace["gpu_id"] = None
            self.namespace["device"] = torch.device("cpu")

        print(f"Worker {rank} initialized")

    def run(self):
        """
        Main worker loop for processing commands.
        
        This method:
        1. Receives messages from the coordinator
        2. Processes different message types:
           - shutdown: Clean shutdown of the worker
           - execute: Execute Python code with streaming output
           - get_var: Retrieve a variable from namespace
           - set_var: Set a variable in namespace
           - sync: Synchronize with other workers
           - get_status: Get worker status
           - get_namespace_info: Get namespace information
        3. Sends results back to coordinator
        4. Handles errors and exceptions
        
        The loop continues until a shutdown message is received.
        """
        while True:
            try:
                message_data = self.socket.recv()
                message = pickle.loads(message_data)

                if message.msg_type == "shutdown":
                    break
                elif message.msg_type == "execute":
                    result = self._execute_code_streaming(message.data, message.msg_id)
                elif message.msg_type == "get_var":
                    result = self._get_variable(message.data)
                elif message.msg_type == "set_var":
                    result = self._set_variable(message.data)
                elif message.msg_type == "sync":
                    dist.barrier()
                    result = {"status": "synced"}
                elif message.msg_type == "get_status":
                    result = self._get_status()
                elif message.msg_type == "get_namespace_info":
                    result = self._get_namespace_info()
                else:
                    result = {"error": f"Unknown message type: {message.msg_type}"}

                # Send response
                response = pickle.dumps(
                    Message(
                        msg_id=message.msg_id,
                        msg_type="response",
                        rank=self.rank,
                        data=result,
                        timestamp=time.time(),
                    )
                )
                self.socket.send(response)

            except Exception as e:
                error_result = {"error": str(e), "traceback": traceback.format_exc()}
                response = pickle.dumps(
                    Message(
                        msg_id=message.msg_id,
                        msg_type="response",
                        rank=self.rank,
                        data=error_result,
                        timestamp=time.time(),
                    )
                )
                self.socket.send(response)

    def _execute_code_streaming(self, code: str, msg_id: str) -> Dict[str, Any]:
        """
        Execute Python code with streaming output support.
        
        This method provides Jupyter-like execution with real-time output where:
        1. Output is streamed to coordinator as it's generated
        2. The last expression's value is captured and returned
        3. Print statements appear immediately during execution
        4. The namespace is preserved between executions
        5. Errors are caught and formatted appropriately
        
        Args:
            code (str): Python code to execute
            msg_id (str): Message ID for tracking streaming output
            
        Returns:
            Dict[str, Any]: Execution result containing:
                - output: Final captured output
                - status: Execution status
                - rank: Worker rank
                - error: Error message if execution failed
                - traceback: Stack trace if execution failed
        """
        try:
            import ast

            # Set up streaming output capture
            old_stdout = sys.stdout
            streaming_output = StreamingOutput(self.socket, self.rank, msg_id)
            sys.stdout = streaming_output

            # First, try to parse the entire code as an expression
            try:
                # Try to parse as a single expression first
                tree = ast.parse(code.strip(), mode='eval')
                # If it's a single expression, evaluate it and capture the result
                result = eval(compile(tree, '<string>', 'eval'), self.namespace)
                
                # Restore stdout and get buffered output
                sys.stdout = old_stdout
                output = streaming_output.getvalue()
                
                # Format the result like Jupyter does
                if result is not None:
                    result_str = repr(result)
                    # Send the result immediately
                    try:
                        result_message = Message(
                            msg_id=f"{msg_id}_result_{time.time()}",
                            msg_type="stream_output",
                            rank=self.rank,
                            data={"text": result_str, "stream": "result"},
                            timestamp=time.time()
                        )
                        self.socket.send(pickle.dumps(result_message))
                    except Exception:
                        pass
                    
                    if output.strip():
                        full_output = output + result_str
                    else:
                        full_output = result_str
                else:
                    full_output = output
                
                # Only send string representation, not the actual object (to avoid pickle issues)
                return {"output": full_output, "status": "success", "rank": self.rank}
                
            except SyntaxError:
                # Not a single expression, parse as statements
                try:
                    tree = ast.parse(code, mode='exec')
                    
                    # Check if the last node is an expression statement
                    if (tree.body and 
                        isinstance(tree.body[-1], ast.Expr)):
                        
                        # Split the AST: execute all but the last statement, then evaluate the last expression
                        if len(tree.body) > 1:
                            # Execute all statements except the last one
                            statements_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                            exec(compile(statements_tree, '<string>', 'exec'), self.namespace)
                        
                        # Evaluate the last expression
                        last_expr = tree.body[-1].value  # Get the expression from the Expr node
                        result = eval(compile(ast.Expression(body=last_expr), '<string>', 'eval'), self.namespace)
                        
                        # Restore stdout and get buffered output
                        sys.stdout = old_stdout
                        output = streaming_output.getvalue()
                        
                        # Format the result
                        if result is not None:
                            result_str = repr(result)
                            # Send the result immediately
                            try:
                                result_message = Message(
                                    msg_id=f"{msg_id}_result_{time.time()}",
                                    msg_type="stream_output",
                                    rank=self.rank,
                                    data={"text": result_str, "stream": "result"},
                                    timestamp=time.time()
                                )
                                self.socket.send(pickle.dumps(result_message))
                            except Exception:
                                pass
                            
                            if output.strip():
                                full_output = output + result_str
                            else:
                                full_output = result_str
                        else:
                            full_output = output
                            
                        # Only send string representation, not the actual object (to avoid pickle issues)
                        return {"output": full_output, "status": "success", "rank": self.rank}
                    
                    else:
                        # Last statement is not an expression, execute normally
                        exec(compile(tree, '<string>', 'exec'), self.namespace)
                        
                        # Restore stdout and get buffered output
                        sys.stdout = old_stdout
                        output = streaming_output.getvalue()
                        
                        return {"output": output, "status": "success", "rank": self.rank}
                        
                except SyntaxError as e:
                    # If we can't parse it at all, restore stdout and raise the error
                    sys.stdout = old_stdout
                    raise e

        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = old_stdout
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "rank": self.rank,
            }

    def _get_variable(self, var_name: str) -> Dict[str, Any]:
        """
        Retrieve a variable from the worker's namespace.
        
        This method handles special cases for different types:
        - PyTorch tensors: Returns device, dtype, and shape info
        - Regular Python objects: Returns pickled value
        
        Args:
            var_name (str): Name of variable to retrieve
            
        Returns:
            Dict[str, Any]: Variable information containing:
                - value: The variable value
                - device: Device info for tensors
                - dtype: Data type for tensors
                - shape: Shape for tensors
                - error: Error message if retrieval failed
        """
        try:
            if var_name in self.namespace:
                value = self.namespace[var_name]
                # Handle torch tensors specially
                if isinstance(value, torch.Tensor):
                    return {
                        "value": value.cpu().detach(),
                        "device": str(value.device),
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                    }
                else:
                    return {"value": value}
            else:
                return {"error": f"Variable {var_name} not found"}
        except Exception as e:
            return {"error": str(e)}

    def _get_namespace_info(self) -> Dict[str, Any]:
        """Get type and signature information for all variables in namespace for IDE integration"""
        try:
            import inspect
            import types
            
            namespace_info = {}
            
            for name, obj in self.namespace.items():
                if name.startswith('_'):  # Skip private variables
                    continue
                    
                info = {
                    "name": name,
                    "type": type(obj).__name__,
                    "module": getattr(type(obj), '__module__', None)
                }
                
                # Add specific information based on object type
                if isinstance(obj, torch.Tensor):
                    info.update({
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "device": str(obj.device),
                        "tensor_type": "torch.Tensor"
                    })
                elif isinstance(obj, torch.device):
                    info.update({
                        "device_type": str(obj.type),
                        "device_index": obj.index,
                        "torch_device": True
                    })
                elif callable(obj):
                    try:
                        sig = inspect.signature(obj)
                        info.update({
                            "signature": str(sig),
                            "callable": True,
                            "doc": inspect.getdoc(obj)
                        })
                    except (ValueError, TypeError):
                        info["callable"] = True
                elif isinstance(obj, types.ModuleType):
                    info.update({
                        "module_name": obj.__name__,
                        "module_file": getattr(obj, '__file__', None),
                        "is_module": True
                    })
                elif hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
                    info.update({
                        "class_name": obj.__class__.__name__,
                        "repr": repr(obj) if len(repr(obj)) < 200 else f"{repr(obj)[:200]}..."
                    })
                
                namespace_info[name] = info
                
            return {"namespace_info": namespace_info, "status": "success"}
            
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _set_variable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set a variable in the worker's namespace.
        
        Args:
            data (Dict[str, Any]): Dictionary containing:
                - name: Variable name
                - value: Variable value
                
        Returns:
            Dict[str, Any]: Result of operation:
                - status: Success/failure
                - error: Error message if failed
        """
        try:
            name = data["name"]
            value = data["value"]
            self.namespace[name] = value
            return {"status": "success"}
        except Exception as e:
            return {"error": str(e)}

    def _get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about this worker.
        
        Returns information about:
        - Process status
        - GPU assignment and utilization
        - Memory usage
        - CUDA availability
        - Current device
        
        Returns:
            Dict[str, Any]: Status information containing:
                - pid: Process ID
                - running: Whether process is running
                - gpu_id: Assigned GPU ID
                - gpu_name: GPU device name
                - gpu_memory_allocated: Allocated GPU memory
                - gpu_memory_reserved: Reserved GPU memory
                - gpu_memory_total: Total GPU memory
                - cuda_available: Whether CUDA is available
                - current_device: Current device (CPU/GPU)
        """
        status = {
            "rank": self.rank,
            "world_size": self.world_size,
            "gpu_id": self.gpu_id,
        }

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            status.update(
                {
                    "cuda_available": True,
                    "current_device": current_device,
                    "gpu_name": torch.cuda.get_device_name(current_device),
                    "gpu_memory_allocated": torch.cuda.memory_allocated(current_device)
                    / 1024**3,  # GB
                    "gpu_memory_reserved": torch.cuda.memory_reserved(current_device)
                    / 1024**3,  # GB
                    "gpu_memory_total": torch.cuda.get_device_properties(
                        current_device
                    ).total_memory
                    / 1024**3,  # GB
                }
            )
        else:
            status.update(
                {
                    "cuda_available": False,
                    "current_device": "cpu",
                    "gpu_name": "CPU",
                    "gpu_memory_allocated": 0,
                    "gpu_memory_reserved": 0,
                    "gpu_memory_total": 0,
                }
            )

        return status

    def shutdown(self):
        """
        Clean shutdown of the worker process.
        
        This method:
        1. Closes ZMQ socket and context
        2. Cleans up PyTorch distributed
        3. Releases GPU resources
        """
        dist.destroy_process_group()
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    master_addr = sys.argv[3]
    master_port = sys.argv[4]
    comm_port = int(sys.argv[5])

    # GPU ID is optional (6th argument)
    gpu_id = None
    if len(sys.argv) > 6:
        gpu_id = int(sys.argv[6])

    worker = DistributedWorker(
        rank, world_size, master_addr, master_port, comm_port, gpu_id
    )
    try:
        worker.run()
    finally:
        worker.shutdown()
