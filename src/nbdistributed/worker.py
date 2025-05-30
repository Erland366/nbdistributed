"""
Worker process for distributed execution
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
from nbdistributed.communication import Message


class DistributedWorker:
    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        comm_port: int,
        gpu_id: Optional[int] = None,
    ):
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
        """Main worker loop"""
        while True:
            try:
                message_data = self.socket.recv()
                message = pickle.loads(message_data)

                if message.msg_type == "shutdown":
                    break
                elif message.msg_type == "execute":
                    result = self._execute_code(message.data)
                elif message.msg_type == "get_var":
                    result = self._get_variable(message.data)
                elif message.msg_type == "set_var":
                    result = self._set_variable(message.data)
                elif message.msg_type == "sync":
                    dist.barrier()
                    result = {"status": "synced"}
                elif message.msg_type == "get_status":
                    result = self._get_status()
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

    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code in worker namespace with REPL-like behavior"""
        try:
            # Capture stdout
            from io import StringIO
            import ast

            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # First, try to parse the entire code as an expression
            try:
                # Try to parse as a single expression first
                tree = ast.parse(code.strip(), mode='eval')
                # If it's a single expression, evaluate it and capture the result
                result = eval(compile(tree, '<string>', 'eval'), self.namespace)
                
                # Restore stdout and get output
                sys.stdout = old_stdout
                output = captured_output.getvalue()
                
                # Format the result like Jupyter does
                if result is not None:
                    result_str = repr(result)
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
                        
                        # Restore stdout and get output
                        sys.stdout = old_stdout
                        output = captured_output.getvalue()
                        
                        # Format the result
                        if result is not None:
                            result_str = repr(result)
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
                        
                        # Restore stdout and get output
                        sys.stdout = old_stdout
                        output = captured_output.getvalue()
                        
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
        """Get variable from namespace"""
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

    def _set_variable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Set variable in namespace"""
        try:
            self.namespace[data["name"]] = data["value"]
            return {"status": "success"}
        except Exception as e:
            return {"error": str(e)}

    def _get_status(self) -> Dict[str, Any]:
        """Get detailed status information including GPU details"""
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
        """Cleanup worker"""
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
