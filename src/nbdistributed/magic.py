# jupyter_distributed/magic.py
"""
IPython magic commands for distributed execution
"""

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from typing import Optional, List, Dict, Any

from nbdistributed.process_manager import ProcessManager
from nbdistributed.communication import CommunicationManager


@magics_class
class DistributedMagic(Magics):
    _process_manager: Optional[ProcessManager] = None
    _comm_manager: Optional[CommunicationManager] = None
    _num_processes: int = 0

    @line_magic
    @magic_arguments()
    @argument(
        "--num-processes", "-n", type=int, default=2, help="Number of GPUs to utilize"
    )
    @argument(
        "--master-addr", "-a", type=str, default="localhost", help="Master address"
    )
    @argument(
        "--gpu-ids",
        "-g",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,3'). If not specified, cycles through all available GPUs.",
    )
    def dist_init(self, line):
        """Initialize distributed workers"""
        args = parse_argstring(self.dist_init, line)

        if self._process_manager and self._process_manager.is_running():
            print(
                "Distributed workers already running. Use %dist_shutdown to stop them first."
            )
            return

        try:
            # Parse GPU IDs if provided
            gpu_ids = None
            if args.gpu_ids:
                try:
                    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
                    print(f"Using GPU IDs: {gpu_ids}")

                    # Validate GPU IDs
                    import torch

                    if torch.cuda.is_available():
                        available_gpus = list(range(torch.cuda.device_count()))
                        invalid_gpus = [
                            gpu_id for gpu_id in gpu_ids if gpu_id not in available_gpus
                        ]
                        if invalid_gpus:
                            print(f"❌ Invalid GPU IDs: {invalid_gpus}")
                            print(f"Available GPUs: {available_gpus}")
                            return

                        if len(gpu_ids) < args.num_processes:
                            print(
                                f"❌ Not enough GPU IDs specified. Need {args.num_processes}, got {len(gpu_ids)}"
                            )
                            print(
                                "Either specify more GPU IDs or reduce --num-processes"
                            )
                            return
                    else:
                        print("⚠️  CUDA not available, GPU IDs will be ignored")
                        gpu_ids = None

                except ValueError:
                    print(
                        "❌ Invalid GPU IDs format. Use comma-separated integers (e.g., '0,1,3')"
                    )
                    return

            print(f"Starting {args.num_processes} distributed workers...")

            # Start process manager
            self._process_manager = ProcessManager()
            comm_port = self._process_manager.start_workers(
                args.num_processes, args.master_addr, gpu_ids
            )

            # Start communication manager
            self._comm_manager = CommunicationManager(args.num_processes, comm_port)
            self._num_processes = args.num_processes

            print(f"✓ Successfully started {args.num_processes} workers")
            if gpu_ids:
                for rank, gpu_id in enumerate(gpu_ids[: args.num_processes]):
                    print(f"  Rank {rank} -> GPU {gpu_id}")

            print("Available commands:")
            print("  %%distributed - Execute code on all ranks")
            print("  %%rank[0,1] - Execute code on specific ranks")
            print("  %sync - Synchronize all ranks")
            print("  %dist_status - Show worker status")
            print("  %dist_shutdown - Shutdown workers")

        except Exception as e:
            print(f"Failed to start distributed workers: {e}")
            self.shutdown_all()

    @line_magic
    def dist_status(self, line):
        """Show status of distributed workers"""
        if not self._process_manager:
            print("No distributed workers running")
            return

        print(f"Distributed cluster status ({self._num_processes} processes):")
        print("=" * 60)

        # Get detailed status including GPU information
        status = self._process_manager.get_detailed_status(self._comm_manager)

        for rank in sorted(status.keys()):
            info = status[rank]
            status_emoji = "✓" if info["running"] else "✗"

            # Basic info
            print(f"Rank {rank}: {status_emoji} PID {info['pid']}")

            # GPU assignment info
            if info.get("gpu_id") is not None:
                gpu_name = info.get("gpu_name", f"GPU {info['gpu_id']}")
                print(f"  ├─ GPU: {info['gpu_id']} ({gpu_name})")

                # Memory info if available from live worker
                if "gpu_memory_total" in info and info["gpu_memory_total"] > 0:
                    allocated = info.get("gpu_memory_allocated", 0)
                    reserved = info.get("gpu_memory_reserved", 0)
                    total = info.get("gpu_memory_total", 0)
                    utilization = (allocated / total * 100) if total > 0 else 0
                    print(
                        f"  ├─ Memory: {allocated:.1f}GB / {total:.1f}GB ({utilization:.1f}% used)"
                    )
                    if reserved > allocated:
                        print(f"  ├─ Reserved: {reserved:.1f}GB")
            else:
                device_name = info.get("gpu_name", "CPU")
                print(f"  ├─ Device: {device_name}")

            # Worker status
            if info["running"]:
                print("  └─ Status: Running")
            else:
                print(
                    f"  └─ Status: Stopped (exit code: {info.get('returncode', 'unknown')})"
                )

            print()  # Empty line between ranks

    @line_magic
    @magic_arguments()
    @argument(
        "--force",
        "-f",
        action="store_true",
        help="Force shutdown even if communication fails",
    )
    def dist_shutdown(self, line):
        """Shutdown distributed workers"""
        args = parse_argstring(self.dist_shutdown, line)

        if args.force:
            print("Force shutting down distributed workers...")
            self.force_shutdown_all()
        else:
            self.shutdown_all()
        print("Distributed workers shutdown")

    @classmethod
    def force_shutdown_all(cls):
        """Force shutdown all distributed components without waiting for responses"""
        print("Starting force shutdown...")

        # First try graceful shutdown of communication
        if cls._comm_manager:
            try:
                print("Sending shutdown signal to workers...")
                cls._comm_manager.send_to_all("shutdown", {}, timeout=2.0)
                print("Shutdown signal sent successfully")
            except Exception as e:
                print(f"Failed to send shutdown signal: {e}")
            try:
                print("Shutting down communication manager...")
                cls._comm_manager.shutdown()
                print("Communication manager shut down")
            except Exception as e:
                print(f"Failed to shutdown communication manager: {e}")
            cls._comm_manager = None

        # Always use nuclear shutdown for processes since it's most reliable
        if cls._process_manager:
            print("Using nuclear shutdown for processes...")
            cls._nuclear_shutdown()
            cls._process_manager = None

        cls._num_processes = 0
        print("Force shutdown completed")

    @classmethod
    def _nuclear_shutdown(cls):
        """Nuclear option: kill processes using system commands"""
        if not cls._process_manager:
            return

        import os
        import signal

        print("Nuclear shutdown: killing processes directly...")
        for i, process in enumerate(cls._process_manager.processes):
            try:
                pid = process.pid
                print(f"Killing worker {i} with PID {pid}")

                # Try SIGTERM first
                os.kill(pid, signal.SIGTERM)

                # Wait a bit then check if it's dead
                import time

                time.sleep(1)

                # If still alive, use SIGKILL
                try:
                    os.kill(pid, 0)  # Check if process exists
                    print(f"Worker {i} still alive, using SIGKILL")
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    print(f"Worker {i} successfully terminated")

            except ProcessLookupError:
                print(f"Worker {i} already dead")
            except Exception as e:
                print(f"Failed to kill worker {i}: {e}")

        # Clear the process list
        cls._process_manager.processes.clear()
        cls._process_manager.num_ranks = 0
        cls._process_manager.gpu_assignments.clear()
        print("Nuclear shutdown completed")

    @line_magic
    @magic_arguments()
    @argument(
        "--nuclear",
        "-n",
        action="store_true",
        help="Nuclear shutdown: kill all processes directly",
    )
    def dist_reset(self, line):
        """Complete reset of distributed environment using direct process termination"""
        print("=== DISTRIBUTED ENVIRONMENT RESET ===")

        # Always use nuclear shutdown since it's the most reliable
        if self._process_manager:
            print("Performing nuclear shutdown...")
            self._nuclear_shutdown()

        # Force clear everything
        self._comm_manager = None
        self._process_manager = None
        self._num_processes = 0

        print("All state cleared")
        print("You can now run %dist_init to start fresh")
        print("=======================================")

    @classmethod
    def shutdown_all(cls):
        """Shutdown all distributed components"""
        if cls._comm_manager:
            try:
                cls._comm_manager.send_to_all("shutdown", {}, timeout=5.0)
            except Exception as e:
                print(f"Warning: Could not send shutdown signal to workers: {e}")
            try:
                cls._comm_manager.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down communication manager: {e}")
            cls._comm_manager = None

        if cls._process_manager:
            try:
                cls._process_manager.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down process manager: {e}")
            cls._process_manager = None

        cls._num_processes = 0

    @cell_magic
    def distributed(self, line, cell):
        """Execute code on all ranks"""
        if not self._comm_manager:
            print("No distributed workers running. Use %dist_init first.")
            return

        try:
            responses = self._comm_manager.send_to_all("execute", cell)
            self._display_responses(responses, "All ranks")
        except Exception as e:
            print(f"Error executing distributed code: {e}")

    @cell_magic
    def rank(self, line, cell):
        """Execute code on specific ranks"""
        if not self._comm_manager:
            print("No distributed workers running. Use %dist_init first.")
            return

        # Parse rank specification
        ranks = self._parse_ranks(line)
        if not ranks:
            print("Invalid rank specification. Use: %%rank[0,1,2] or %%rank[0-2]")
            return

        try:
            responses = self._comm_manager.send_to_ranks(ranks, "execute", cell)
            self._display_responses(responses, f"Ranks {ranks}")
        except Exception as e:
            print(f"Error executing code on ranks {ranks}: {e}")

    @line_magic
    def sync(self, line):
        """Synchronize all ranks"""
        if not self._comm_manager:
            print("No distributed workers running. Use %dist_init first.")
            return

        try:
            responses = self._comm_manager.send_to_all("sync", {})
            print(f"✓ Synchronized {len(responses)} ranks")
        except Exception as e:
            print(f"Error synchronizing ranks: {e}")

    @line_magic
    def dist_debug(self, line):
        """Debug information about the distributed state"""
        print("=== Distributed Debug Information ===")
        print(f"Process manager exists: {self._process_manager is not None}")
        print(f"Communication manager exists: {self._comm_manager is not None}")
        print(f"Number of processes: {self._num_processes}")

        if self._process_manager:
            print(f"Process manager is_running(): {self._process_manager.is_running()}")
            print(
                f"Number of processes tracked: {len(self._process_manager.processes)}"
            )

            for i, process in enumerate(self._process_manager.processes):
                poll_result = process.poll()
                status = (
                    "Running"
                    if poll_result is None
                    else f"Dead (exit code: {poll_result})"
                )
                print(f"  Process {i} (PID: {process.pid}): {status}")

        print("=====================================")

    def _parse_ranks(self, line: str) -> List[int]:
        """Parse rank specification like [0,1,2] or [0-2]"""
        line = line.strip()
        if not line.startswith("[") or not line.endswith("]"):
            return []

        rank_spec = line[1:-1]
        ranks = []

        for part in rank_spec.split(","):
            part = part.strip()
            if "-" in part:
                # Range specification like 0-2
                start, end = map(int, part.split("-"))
                ranks.extend(range(start, end + 1))
            else:
                # Single rank
                ranks.append(int(part))

        # Filter valid ranks
        return [r for r in ranks if 0 <= r < self._num_processes]

    def _display_responses(self, responses: Dict[int, Any], title: str):
        """Display responses from workers"""
        print(f"\n=== {title} ===")

        for rank in sorted(responses.keys()):
            response = responses[rank]
            print(f"\n--- Rank {rank} ---")

            if "error" in response:
                print(f"❌ Error: {response['error']}")
                if "traceback" in response:
                    print(response["traceback"])
            else:
                if response.get("output"):
                    print(response["output"])
                else:
                    print("✓ Executed successfully")
