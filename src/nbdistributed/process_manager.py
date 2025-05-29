# jupyter_distributed/process_manager.py
"""
Process management for distributed workers
"""

import subprocess
import time
import os
from typing import List, Optional
import socket


class ProcessManager:
    def __init__(self):
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
        """Start distributed worker processes

        Args:
            num_processes: Number of worker processes to start
            master_addr: Master node address
            gpu_ids: Optional list of GPU IDs to assign to workers. If None, cycles through all available GPUs.
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
        """Find a free port"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def shutdown(self):
        """Shutdown all worker processes"""
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
        """Check if workers are still running"""
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
        """Get status of all workers including GPU information"""
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
        """Get GPU name for a given GPU ID"""
        try:
            import torch

            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                return torch.cuda.get_device_name(gpu_id)
            else:
                return f"GPU {gpu_id} (unavailable)"
        except Exception:
            return f"GPU {gpu_id} (unknown)"

    def get_detailed_status(self, comm_manager=None) -> dict:
        """Get detailed status including live GPU information from workers"""
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
