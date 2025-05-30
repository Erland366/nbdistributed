# jupyter_distributed/process_manager.py
"""
Process management for distributed PyTorch workers in Jupyter notebooks.

This module handles the lifecycle of distributed worker processes, including:
- Starting worker processes with appropriate GPU assignments
- Managing process state and communication ports
- Monitoring process health
- Graceful shutdown and cleanup
- Status reporting with GPU information

The ProcessManager works in conjunction with the CommunicationManager to provide
a robust distributed execution environment for Jupyter notebooks.
"""

import subprocess
import time
import os
from typing import List, Optional
import socket


class ProcessManager:
    """
    Manager for distributed PyTorch worker processes.
    
    This class handles the lifecycle of worker processes that execute distributed
    PyTorch code. It manages:
    - Process creation and initialization
    - GPU assignments
    - Port allocation
    - Process monitoring
    - Graceful shutdown
    - Status reporting
    
    Attributes:
        processes (List[subprocess.Popen]): List of worker processes
        num_processes (int): Number of active worker processes
        master_port (Optional[int]): Port for PyTorch distributed communication
        comm_port (Optional[int]): Port for ZMQ communication
        gpu_assignments (dict): Mapping of rank to GPU ID
    """

    def __init__(self):
        """
        Initialize the process manager.
        
        Creates an empty process manager with no active workers.
        All ports and GPU assignments will be set when workers are started.
        """
        self.processes: List[subprocess.Popen] = []
        self.num_processes = 0
        self.master_port = None
        self.comm_port = None
        self.gpu_assignments = {}  # Track GPU assignments per rank

    def start_workers(
        self,
        num_processes: int,
        master_addr: str = "localhost",
        gpu_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Start distributed worker processes.
        
        This method:
        1. Finds available ports for communication
        2. Assigns GPUs to workers (if available)
        3. Starts worker processes
        4. Verifies successful startup
        
        Args:
            num_processes (int): Number of worker processes to start
            master_addr (str): Master node address (default: "localhost")
            gpu_ids (Optional[List[int]]): Specific GPU IDs to assign to workers.
                If None, cycles through all available GPUs.
                
        Returns:
            int: Port number for ZMQ communication
            
        Raises:
            RuntimeError: If any worker fails to start
            
        Example:
            >>> manager = ProcessManager()
            >>> comm_port = manager.start_workers(4, gpu_ids=[0,1,2,3])
            Starting 4 worker processes...
            Worker 0 using GPU 0
            Worker 1 using GPU 1
            Worker 2 using GPU 2
            Worker 3 using GPU 3
        """
        self.num_processes = num_processes
        self.gpu_assignments = {}

        # Find available ports
        self.master_port = self._find_free_port()
        self.comm_port = self._find_free_port()

        # Get path to worker script
        worker_script = os.path.join(os.path.dirname(__file__), "worker.py")

        # Start worker processes
        for rank in range(num_processes):
            # Determine GPU ID for this rank
            gpu_id = None
            if gpu_ids:
                gpu_id = (
                    gpu_ids[rank]
                    if rank < len(gpu_ids)
                    else gpu_ids[rank % len(gpu_ids)]
                )

            # Store GPU assignment
            self.gpu_assignments[rank] = gpu_id

            cmd = [
                "python",
                worker_script,
                str(rank),
                str(num_processes),
                master_addr,
                str(self.master_port),
                str(self.comm_port),
            ]

            # Add GPU ID if specified
            if gpu_id is not None:
                cmd.append(str(gpu_id))

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            self.processes.append(process)

        # Wait a bit for processes to initialize
        time.sleep(2)

        # Check if all processes started successfully
        for i, process in enumerate(self.processes):
            if process.poll() is not None:
                # Process died, clean up and show error
                stdout, stderr = process.communicate()
                self.shutdown()
                error_msg = f"Worker {i} failed to start"
                if stderr:
                    error_msg += f"\nSTDERR: {stderr}"
                if stdout:
                    error_msg += f"\nSTDOUT: {stdout}"
                raise RuntimeError(error_msg)

        return self.comm_port

    def _find_free_port(self) -> int:
        """
        Find an available network port.
        
        This method:
        1. Creates a temporary socket
        2. Binds to port 0 (lets OS choose port)
        3. Gets the assigned port number
        4. Closes the socket
        
        Returns:
            int: Available port number
            
        Note:
            There is a small chance the port could be taken between
            finding it and using it. The caller should handle this case.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def shutdown(self):
        """
        Shutdown all worker processes gracefully.
        
        This method attempts a clean shutdown by:
        1. Sending SIGTERM to each process
        2. Waiting for processes to exit (with timeout)
        3. Force killing (SIGKILL) processes that don't exit
        4. Cleaning up internal state
        
        The shutdown is done in stages:
        - First tries graceful termination (SIGTERM)
        - Waits up to 3 seconds for each process
        - Falls back to force kill (SIGKILL) if needed
        - Waits additional 2 seconds after force kill
        
        Any processes that survive even SIGKILL are logged as warnings.
        """
        print(f"Shutting down {len(self.processes)} worker processes...")

        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Still running
                print(f"Terminating worker {i} (PID: {process.pid})")
                process.terminate()

        # Wait for all processes to terminate
        terminated_count = 0
        for i, process in enumerate(self.processes):
            try:
                print(f"Waiting for worker {i} to terminate...")
                process.wait(timeout=3)
                terminated_count += 1
                print(f"Worker {i} terminated gracefully")
            except subprocess.TimeoutExpired:
                print(f"Worker {i} didn't terminate gracefully, force killing...")
                process.kill()
                try:
                    process.wait(timeout=2)
                    terminated_count += 1
                    print(f"Worker {i} force killed")
                except subprocess.TimeoutExpired:
                    print(f"Warning: Could not kill worker {i} (PID: {process.pid})")

        print(
            f"Successfully shut down {terminated_count}/{len(self.processes)} workers"
        )

        self.processes.clear()
        self.num_processes = 0
        self.gpu_assignments.clear()
        print("Process manager cleanup completed")

    def is_running(self) -> bool:
        """
        Check if any worker processes are still running.
        
        This method:
        1. Returns False if no processes exist
        2. Checks each process's status
        3. Cleans up dead processes from the list
        4. Returns True if any processes are still alive
        
        Returns:
            bool: True if any workers are still running
            
        Note:
            This method has the side effect of cleaning up dead processes
            from the internal process list.
        """
        if not self.processes:
            return False

        # Check each process individually and clean up dead ones
        alive_processes = []
        for process in self.processes:
            if process.poll() is None:  # Still running
                alive_processes.append(process)

        # Update processes list to only include alive ones
        self.processes = alive_processes

        return len(self.processes) > 0

    def get_status(self) -> dict:
        """
        Get basic status information for all workers.
        
        This method returns static information about each worker:
        - Process ID
        - Running state
        - Exit code (if terminated)
        - GPU assignment
        - GPU name (if available)
        
        Returns:
            dict: Status information keyed by worker rank:
                {
                    rank: {
                        "pid": process ID,
                        "running": bool,
                        "returncode": exit code or None,
                        "gpu_id": GPU ID or None,
                        "gpu_name": GPU name or "CPU"
                    }
                }
        """
        status = {}
        for i, process in enumerate(self.processes):
            gpu_id = self.gpu_assignments.get(i)
            gpu_name = self._get_gpu_name(gpu_id) if gpu_id is not None else "CPU"

            status[i] = {
                "pid": process.pid,
                "running": process.poll() is None,
                "returncode": process.returncode,
                "gpu_id": gpu_id,
                "gpu_name": gpu_name,
            }
        return status

    def _get_gpu_name(self, gpu_id: int) -> str:
        """
        Get the name of a GPU device.
        
        Args:
            gpu_id (int): GPU device ID
            
        Returns:
            str: GPU device name if available, otherwise a fallback string:
                - Actual device name (e.g., "NVIDIA A100")
                - "GPU {id} (unavailable)" if GPU exists but can't be accessed
                - "GPU {id} (unknown)" if GPU info can't be retrieved
                
        Note:
            This method gracefully handles cases where:
            - CUDA is not available
            - The GPU ID is invalid
            - torch.cuda fails for any reason
        """
        try:
            import torch

            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                return torch.cuda.get_device_name(gpu_id)
            else:
                return f"GPU {gpu_id} (unavailable)"
        except Exception:
            return f"GPU {gpu_id} (unknown)"

    def get_detailed_status(self, comm_manager=None) -> dict:
        """
        Get detailed status including live information from workers.
        
        This method combines:
        1. Basic process status information
        2. Live GPU utilization from workers (if available)
        3. Additional worker-reported metrics
        
        Args:
            comm_manager: Optional communication manager to get live worker info
            
        Returns:
            dict: Detailed status information keyed by worker rank:
                {
                    rank: {
                        # Basic process info
                        "pid": process ID,
                        "running": bool,
                        "returncode": exit code or None,
                        "gpu_id": GPU ID or None,
                        "gpu_name": GPU name or "CPU",
                        
                        # Live info (if available)
                        "gpu_memory_allocated": float,
                        "gpu_memory_reserved": float,
                        "gpu_memory_total": float,
                        ...
                    }
                }
                
        Note:
            If communication with workers fails, falls back to basic status info.
        """
        status = self.get_status()

        # If we have a communication manager, get live info from workers
        if comm_manager and self.is_running():
            try:
                responses = comm_manager.send_to_all("get_status", {}, timeout=5.0)
                for rank, response in responses.items():
                    if rank in status and "error" not in response:
                        # Update with live information from worker
                        status[rank].update(response)
            except Exception:
                # If communication fails, just use basic status
                pass

        return status
