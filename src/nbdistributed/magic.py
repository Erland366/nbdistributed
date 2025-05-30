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
    _distributed_mode_active: bool = False

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
            print("  %%distributed - Execute code on all ranks (explicit)")
            print("  %%rank[0,1] - Execute code on specific ranks")
            print("  %sync - Synchronize all ranks")
            print("  %dist_status - Show worker status")
            print("  %dist_mode - Toggle automatic distributed mode")
            print("  %dist_shutdown - Shutdown workers")
            print()
            print("🚀 Distributed mode active: All cells will now execute on workers automatically!")
            print("   Magic commands (%, %%) will still execute locally as normal.")

            # Enable automatic distributed execution
            self._enable_distributed_mode()

        except Exception as e:
            print(f"Failed to start distributed workers: {e}")
            self.shutdown_all()

    def _enable_distributed_mode(self):
        """Enable automatic distributed execution using input transformer"""
        if not self._distributed_mode_active:
            self.shell.input_transformers_cleanup.append(self._distributed_transformer)
            self._distributed_mode_active = True

    def _disable_distributed_mode(self):
        """Disable automatic distributed execution"""
        if self._distributed_mode_active:
            try:
                self.shell.input_transformers_cleanup.remove(self._distributed_transformer)
            except ValueError:
                pass  # Already removed
            self._distributed_mode_active = False

    def _distributed_transformer(self, lines):
        """Transform non-magic cells to use %%distributed automatically"""
        if not lines:
            return lines
            
        # Join lines to check the full cell content
        full_cell = '\n'.join(lines)
        stripped = full_cell.strip()
        
        # Don't transform if:
        # 1. It's already a magic command
        # 2. It's empty or whitespace only
        # 3. It's a comment only
        if (not stripped or 
            stripped.startswith('%') or 
            all(line.strip().startswith('#') or not line.strip() for line in lines)):
            return lines
        
        # Transform by prepending %%distributed
        return ['%%distributed'] + lines

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
    def dist_shutdown(self, line):
        """Shutdown distributed workers using nuclear option"""
        print("Shutting down distributed workers (nuclear option)...")
        self.force_shutdown_all()
        
        # Disable distributed mode
        self._disable_distributed_mode()
        
        # CRITICAL: Also clear instance variables since dist_init creates them
        self._process_manager = None
        self._comm_manager = None
        self._num_processes = 0
        
        print("Distributed workers shutdown")
        print("📱 Normal cell execution restored")

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
        """Nuclear option: kill ALL processes related to distributed workers"""
        import os
        import signal
        import subprocess
        import time

        print("🚀 NUCLEAR SHUTDOWN: Terminating ALL related processes...")
        
        # First, kill processes by pattern (any process with worker.py or similar)
        try:
            print("Killing processes by pattern...")
            subprocess.run(["pkill", "-f", "worker.py"], timeout=5)
            subprocess.run(["pkill", "-f", "nbdistributed"], timeout=5) 
            subprocess.run(["pkill", "-f", "distributed.*worker"], timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Pattern kill failed (continuing): {e}")

        # Kill tracked processes and their process groups
        if cls._process_manager and cls._process_manager.processes:
            for i, process in enumerate(cls._process_manager.processes):
                try:
                    pid = process.pid
                    print(f"Nuking worker {i} (PID {pid}) and process group...")
                    
                    # Kill the entire process group first
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        print(f"Process group for worker {i} killed")
                    except (ProcessLookupError, OSError):
                        pass
                    
                    # Then kill the specific process
                    try:
                        os.kill(pid, signal.SIGKILL)
                        print(f"Worker {i} process killed")
                    except ProcessLookupError:
                        print(f"Worker {i} already dead")
                        
                except Exception as e:
                    print(f"Failed to kill worker {i}: {e}")

            # CRITICAL: Force all subprocess objects to update their status
            print("Forcing subprocess objects to recognize process death...")
            for i, process in enumerate(cls._process_manager.processes):
                try:
                    # Force the subprocess to check if it's dead
                    process.poll()
                    # Try to wait with timeout to clean up zombie
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                    print(f"Subprocess {i} status updated")
                except Exception as e:
                    print(f"Error updating subprocess {i}: {e}")

        # Kill any remaining processes that might be hanging around
        try:
            print("Final cleanup: killing any remaining distributed processes...")
            subprocess.run(["pkill", "-9", "-f", "python.*worker"], timeout=5)
            subprocess.run(["pkill", "-9", "-f", "torch.*distributed"], timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Clear all state
        if cls._process_manager:
            cls._process_manager.processes.clear()
            cls._process_manager.num_processes = 0
            cls._process_manager.gpu_assignments.clear()
            
        print("💥 NUCLEAR SHUTDOWN COMPLETED - All processes terminated")

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

        # Disable distributed mode
        self._disable_distributed_mode()

        # Force clear everything
        self._comm_manager = None
        self._process_manager = None
        self._num_processes = 0

        print("All state cleared")
        print("You can now run %dist_init to start fresh")
        print("📱 Normal cell execution restored")
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
        
        # Note: We don't disable distributed mode here since this is a class method
        # and we don't have access to the instance. Individual instances should
        # call _disable_distributed_mode() when they shut down.

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
        print(f"Distributed mode active: {self._distributed_mode_active}")

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

    @line_magic
    @magic_arguments()
    @argument(
        "--enable", "-e", action="store_true", help="Enable distributed mode"
    )
    @argument(
        "--disable", "-d", action="store_true", help="Disable distributed mode"
    )
    def dist_mode(self, line):
        """Toggle distributed mode on/off without affecting workers"""
        args = parse_argstring(self.dist_mode, line)
        
        if not self._comm_manager:
            print("No distributed workers running. Use %dist_init first.")
            return
            
        if args.enable and args.disable:
            print("Cannot specify both --enable and --disable")
            return
            
        if args.enable:
            if not self._distributed_mode_active:
                self._enable_distributed_mode()
                print("🚀 Distributed mode enabled: Regular cells will execute on workers")
            else:
                print("Distributed mode is already enabled")
        elif args.disable:
            if self._distributed_mode_active:
                self._disable_distributed_mode()
                print("📱 Distributed mode disabled: Regular cells will execute locally")
            else:
                print("Distributed mode is already disabled")
        else:
            # Show current status
            status = "enabled" if self._distributed_mode_active else "disabled"
            print(f"Distributed mode is currently {status}")
            print("Use %dist_mode --enable or %dist_mode --disable to toggle")

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
        """Display responses from workers with enhanced REPL-like formatting"""
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
                    # The output now includes both stdout and expression results as a single string
                    print(response["output"])
                else:
                    print("✓ Executed successfully")
