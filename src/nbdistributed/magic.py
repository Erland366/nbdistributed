# jupyter_distributed/magic.py
"""
IPython magic commands for distributed execution in Jupyter notebooks.

This module provides IPython magic commands for distributed execution of Python code
across multiple processes, particularly useful for distributed PyTorch training.
It enables seamless execution of code across multiple GPUs from within Jupyter notebooks.

Key Features:
- Automatic distribution of code execution across multiple processes
- GPU-aware process management
- Transparent communication between processes
- REPL-like output capturing
- Automatic namespace synchronization for IDE support
- Execution timeline tracking for replay capability
"""

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from typing import Optional, List, Dict, Any
import time
import uuid
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import json

from nbdistributed.process_manager import ProcessManager
from nbdistributed.communication import CommunicationManager


@dataclass
class TimelineEvent:
    """Represents a single event in the execution timeline."""
    timestamp: float
    event_type: str  # 'start', 'output', 'end', 'error'
    content: str
    rank: Optional[int] = None
    line_number: Optional[int] = None
    sub_duration: Optional[float] = None  # Duration of this specific event/line


@dataclass 
class CellExecution:
    """Represents the complete execution of a cell."""
    cell_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    events: Optional[List[TimelineEvent]] = None
    cell_content: str = ""
    execution_count: int = 0
    line_durations: Optional[List[float]] = None  # Duration of each line in the cell
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
        if self.line_durations is None:
            self.line_durations = []


@magics_class
class DistributedMagic(Magics):
    """
    IPython magic commands for distributed execution across multiple processes.
    
    This class provides magic commands that enable distributed execution of Python code
    across multiple processes, with special support for PyTorch distributed training.
    It manages process creation, inter-process communication, and automatic code distribution.
    
    Key Magic Commands:
        %dist_init: Initialize distributed workers
        %%distributed: Execute code on all ranks (explicit)
        %%distributed_auto: Execute code on all ranks (automatic)
        %%rank [n]: Execute code on specific ranks
        %sync: Synchronize all ranks
        %dist_status: Show worker status
        %dist_shutdown: Shutdown workers
        %dist_mode: Toggle automatic distributed mode
        %timeline_save: Manually save timeline data to notebook metadata
        %timeline_debug: Show timeline tracking debug information
        %timeline_clear: Clear all timeline data

        
    Class Attributes:
        _process_manager (Optional[ProcessManager]): Manages distributed worker processes
        _comm_manager (Optional[CommunicationManager]): Handles inter-process communication
        _num_processes (int): Number of active distributed processes
        _distributed_mode_active (bool): Whether automatic distributed execution is enabled
        _execution_timelines (Dict[str, CellExecution]): Timeline data for each cell execution
        _timeline_lock (threading.Lock): Thread lock for timeline data access
        _execution_counter (int): Counter for tracking execution order
    """
    
    _process_manager: Optional[ProcessManager] = None
    _comm_manager: Optional[CommunicationManager] = None
    _num_processes: int = 0
    _distributed_mode_active: bool = False
    
    # Timeline tracking attributes - will be initialized per instance
    _execution_timelines: Optional[Dict[str, CellExecution]] = None
    _timeline_lock: Optional[threading.Lock] = None  
    _execution_counter: int = 0
    
    # Streaming output organized by rank for better readability
    _rank_outputs = None  # Dict to collect output by rank
    _last_displayed_counts = None  # Track what we've already displayed per rank

    def __init__(self, shell=None):
        """Initialize the DistributedMagic class with timeline tracking."""
        super().__init__(shell)
        
        # Initialize timeline tracking per instance
        self._execution_timelines = {}
        self._timeline_lock = threading.Lock()
        
        # Set up notebook save hook to write timeline data to metadata
        self._setup_notebook_save_hook()
        
        # Register timeline tracking handlers immediately (independent of distributed mode)
        self._register_timeline_handlers()

    def _register_timeline_handlers(self):
        """Register timeline tracking handlers immediately when the magic class is loaded."""
        if self.shell is not None and hasattr(self.shell, 'events'):
            try:
                self.shell.events.register('pre_run_cell', self._pre_execution_timeline)
                self.shell.events.register('post_run_cell', self._post_execution_sync)
            except Exception:
                pass  # Fail silently if registration fails

    def _setup_notebook_save_hook(self):
        """Set up a hook to save timeline data to notebook metadata on save."""
        try:
            if (self.shell is not None and 
                hasattr(self.shell, 'kernel') and 
                hasattr(self.shell.kernel, 'session')):
                
                # Try to access the notebook manager through the kernel
                kernel = self.shell.kernel
                if hasattr(kernel, 'session') and hasattr(kernel.session, 'session'):
                    # Register a pre-save hook
                    self._register_save_hook()
        except Exception:
            pass  # Fail silently if we can't set up the hook

    def _register_save_hook(self):
        """Register a hook to intercept notebook saves and add timeline metadata."""
        try:
            # Store the save hook function in the shell for later access
            if self.shell is not None and hasattr(self.shell, 'user_ns'):
                # Register a custom save handler
                self.shell.user_ns['_nb_save_timeline_hook'] = self._save_timeline_to_metadata_hook
                
                # Also register with IPython events if available
                if hasattr(self.shell, 'events'):
                    # Note: This is a custom approach since IPython doesn't have direct notebook save events
                    # We'll store the hook and let it be called manually or through other mechanisms
                    pass
        except Exception:
            pass

    def _save_timeline_to_metadata_hook(self):
        """Hook function to save timeline data to notebook metadata."""
        try:
            if not self._execution_timelines:
                return
                
            # Prepare timeline data for serialization
            timeline_metadata = {}
            for cell_id, execution in self._execution_timelines.items():
                timeline_metadata[cell_id] = {
                    'cell_id': execution.cell_id,
                    'start_time': execution.start_time,
                    'end_time': execution.end_time,
                    'total_duration': execution.total_duration,
                    'cell_content': execution.cell_content,
                    'execution_count': execution.execution_count,
                    'line_durations': execution.line_durations or [],
                    'events': [
                        {
                            'timestamp': event.timestamp,
                            'event_type': event.event_type,
                            'content': event.content,
                            'rank': event.rank,
                            'line_number': event.line_number,
                            'sub_duration': event.sub_duration
                        }
                        for event in (execution.events or [])
                    ]
                }
            
            # Store timeline data in a way that can be accessed for notebook saving
            if self.shell is not None and hasattr(self.shell, 'user_ns'):
                self.shell.user_ns['_notebook_execution_timelines'] = timeline_metadata
                
                # Use IPython's display system to execute JavaScript that modifies notebook metadata
                try:
                    from IPython.display import display, Javascript
                    
                    # Create JavaScript that updates notebook metadata
                    js_code = f"""
                    // Update notebook metadata with timeline data
                    if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {{
                        // Initialize execution_timelines if it doesn't exist
                        if (!Jupyter.notebook.metadata.execution_timelines) {{
                            Jupyter.notebook.metadata.execution_timelines = {{}};
                        }}
                        
                        // Merge the new timeline data
                        const newTimelines = {json.dumps(timeline_metadata)};
                        Object.assign(Jupyter.notebook.metadata.execution_timelines, newTimelines);
                        
                        // Mark notebook as dirty so it gets saved
                        Jupyter.notebook.set_dirty(true);
                        
                        // Log success (can be seen in browser console)
                        console.log('Timeline metadata updated:', Object.keys(newTimelines));
                    }} else {{
                        console.log('Jupyter notebook not available for metadata update');
                    }}
                    """
                    
                    # Display the JavaScript - this will execute in the browser
                    # We suppress output by using element.remove() at the end
                    js_code_with_cleanup = f"""
                    {js_code}
                    
                    // Remove the script element to avoid showing output
                    element.remove();
                    """
                    
                    display(Javascript(js_code_with_cleanup))
                    
                except Exception as e:
                    # Fallback: store in a variable that can be manually saved
                    self.shell.user_ns['_timeline_metadata_for_save'] = timeline_metadata
                    
        except Exception:
            pass  # Fail silently

    def _save_timeline_to_notebook_metadata(self, cell_execution: CellExecution):
        """
        Save timeline data directly to the notebook metadata.
        
        Args:
            cell_execution (CellExecution): The execution timeline to save
        """
        try:
            # Convert timeline to serializable format
            timeline_data = {
                'cell_id': cell_execution.cell_id,
                'start_time': cell_execution.start_time,
                'end_time': cell_execution.end_time,
                'total_duration': cell_execution.total_duration,
                'cell_content': cell_execution.cell_content,
                'execution_count': cell_execution.execution_count,
                'events': [
                    {
                        'timestamp': event.timestamp,
                        'event_type': event.event_type,
                        'content': event.content,
                        'rank': event.rank,
                        'line_number': event.line_number
                    }
                    for event in (cell_execution.events or [])
                ]
            }
            
            # Store timeline data with null checks
            if (self.shell is not None and 
                hasattr(self.shell, 'user_ns') and 
                self.shell.user_ns is not None):
                
                self.shell.user_ns.setdefault('_nb_execution_timelines', {})
                self.shell.user_ns['_nb_execution_timelines'][cell_execution.cell_id] = timeline_data
                
                # Trigger the save hook to update notebook metadata
                self._save_timeline_to_metadata_hook()
                    
        except Exception as e:
            # Don't fail execution if metadata saving fails
            pass

    def _start_cell_execution(self, cell_content: str, execution_count: Optional[int] = None) -> str:
        """
        Start tracking execution timeline for a cell.
        
        Args:
            cell_content (str): The content of the cell being executed
            execution_count (Optional[int]): The actual execution count from the notebook
            
        Returns:
            str: Unique cell execution ID
        """
        if self._timeline_lock is None:
            return ""
            
        with self._timeline_lock:
            # Use provided execution count or increment our counter
            if execution_count is not None:
                actual_execution_count = execution_count
                # Update our counter to match
                self._execution_counter = max(self._execution_counter, execution_count)
            else:
                self._execution_counter += 1
                actual_execution_count = self._execution_counter
            
            cell_id = f"cell_{actual_execution_count}_{uuid.uuid4().hex[:8]}"
            
            start_time = time.time()
            execution = CellExecution(
                cell_id=cell_id,
                start_time=start_time,
                cell_content=cell_content,
                execution_count=actual_execution_count
            )
            
            # Add start event
            start_event = TimelineEvent(
                timestamp=start_time,
                event_type="start",
                content=f"Cell execution started"
            )
            if execution.events is not None:
                execution.events.append(start_event)
            
            # Store in instance for tracking during execution
            if self._execution_timelines is not None:
                self._execution_timelines[cell_id] = execution
            
        return cell_id

    def _update_execution_count(self, cell_id: str, execution_count: int) -> Optional[str]:
        """
        Update the execution count for a timeline entry.
        
        Args:
            cell_id (str): The cell execution ID
            execution_count (int): The actual execution count from Jupyter
            
        Returns:
            Optional[str]: The updated cell_id if changed, None otherwise
        """
        if self._timeline_lock is None or self._execution_timelines is None:
            return None
            
        with self._timeline_lock:
            if cell_id in self._execution_timelines:
                execution = self._execution_timelines[cell_id]
                execution.execution_count = execution_count
                
                # Update the cell_id to reflect the correct execution count
                new_cell_id = f"cell_{execution_count}_{cell_id.split('_')[-1]}"
                
                # Move the timeline entry to the new key
                if new_cell_id != cell_id:
                    execution.cell_id = new_cell_id
                    self._execution_timelines[new_cell_id] = execution
                    del self._execution_timelines[cell_id]
                    
                    # Update our counter to match
                    self._execution_counter = max(self._execution_counter, execution_count)
                    return new_cell_id
                    
        return None

    def _end_cell_execution(self, cell_id: str):
        """
        End tracking execution timeline for a cell and save to notebook metadata.
        
        Args:
            cell_id (str): The cell execution ID
        """
        if self._timeline_lock is None or self._execution_timelines is None:
            return
            
        with self._timeline_lock:
            if cell_id in self._execution_timelines:
                execution = self._execution_timelines[cell_id]
                end_time = time.time()
                execution.end_time = end_time
                execution.total_duration = end_time - execution.start_time
                
                # Add end event
                end_event = TimelineEvent(
                    timestamp=end_time,
                    event_type="end",
                    content=f"Cell execution completed in {execution.total_duration:.4f}s"
                )
                if execution.events is not None:
                    execution.events.append(end_event)
                
                # Save to notebook metadata
                self._save_timeline_to_notebook_metadata(execution)

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
        """
        Initialize distributed workers for parallel execution.
        
        This magic command starts the distributed execution environment by:
        1. Creating worker processes
        2. Assigning GPUs to workers
        3. Setting up communication channels
        4. Enabling automatic distributed execution
        
        Args:
            line (str): Command line arguments:
                --num-processes/-n: Number of worker processes (default: 2)
                --master-addr/-a: Master node address (default: localhost)
                --gpu-ids/-g: Specific GPU IDs to use (e.g., "0,1,3")
                
        Example:
            >>> %dist_init -n 4 -g "0,1,2,3"
            Starting 4 distributed workers...
            ✓ Successfully started 4 workers
            Rank 0 -> GPU 0
            Rank 1 -> GPU 1
            Rank 2 -> GPU 2
            Rank 3 -> GPU 3
        """
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
                            error_msg = f"Not enough GPU IDs specified. Need {args.num_processes}, got {len(gpu_ids)}"
                            print(f"❌ {error_msg}")
                            print(
                                "Either specify more GPU IDs or reduce --num-processes"
                            )
                            return
                    else:
                        print("⚠️  CUDA not available, GPU IDs will be ignored")
                        gpu_ids = None

                except ValueError:
                    error_msg = "Invalid GPU IDs format. Use comma-separated integers (e.g., '0,1,3')"
                    print(f"❌ {error_msg}")
                    return

            print(f"Starting {args.num_processes} distributed workers...")

            # Start process manager
            self._process_manager = ProcessManager()
            comm_port = self._process_manager.start_workers(
                args.num_processes, args.master_addr, gpu_ids
            )

            # Start communication manager with streaming output callback
            self._comm_manager = CommunicationManager(
                args.num_processes, 
                comm_port, 
                output_callback=self._handle_streaming_output
            )
            self._num_processes = args.num_processes

            print(f"✓ Successfully started {args.num_processes} workers")
            if gpu_ids:
                for rank, gpu_id in enumerate(gpu_ids[: args.num_processes]):
                    print(f"  Rank {rank} -> GPU {gpu_id}")

            print("Available commands:")
            print("  %%distributed - Execute code on all ranks (explicit)")
            print("  %%rank [0,n] - Execute code on specific ranks")
            print("  %sync - Synchronize all ranks")
            print("  %dist_status - Show worker status")
            print("  %dist_mode - Toggle automatic distributed mode")
            print("  %dist_shutdown - Shutdown workers")
            print()
            print("🚀 Distributed mode active: All cells will now execute on workers automatically!")
            print("   Magic commands (%, %%) will still execute locally as normal.")
            print()
            print("🐍 Below are auto-imported and special variables auto-generated into the namespace to use")
            print("  `torch`")
            print("  `dist`: `torch.distributed` import alias")
            print("  `rank` (`int`): The local rank")
            print("  `world_size` (`int`): The global world size")
            print("  `gpu_id` (`int`): The specific GPU ID assigned to this worker")
            print("  `device` (`torch.device`): The current PyTorch device object (e.g. `cuda:1`)")

            # Enable automatic distributed execution
            self._enable_distributed_mode()

        except Exception as e:
            print(f"Failed to start distributed workers: {e}")
            self.shutdown_all()

    def _handle_streaming_output(self, rank: int, text: str, stream_type: str):
        """
        Handle real-time streaming output from workers.
        
        Collects output by rank for organized display.
        
        Args:
            rank (int): Worker rank that sent the output
            text (str): Output text
            stream_type (str): Type of stream ("stdout" or "result")
        """
        # Only stream if we have buffers initialized
        if self._rank_outputs is None:
            return
            
        # Filter out system/internal messages that shouldn't be displayed
        if not text or not text.strip():
            return
            
        # Filter out VS Code and Jupyter internal execution tracking
        if any(pattern in text for pattern in [
            'application/vnd.vscode',
            'execution_count',
            'cell_id',
            'msg_id',
            '{"application/',
            "'application/",
        ]):
            return
            
        # Filter out empty dictionaries and system noise, but preserve user data
        text_stripped = text.strip()
        if (text_stripped in ['{}', "{''}"] or 
            text_stripped.startswith("{'application/") or
            text_stripped.startswith('{"application/')):
            return
        
        # Store output by rank (no formatting here - keep it clean)
        if rank not in self._rank_outputs:
            self._rank_outputs[rank] = []
        
        self._rank_outputs[rank].append(text_stripped)

    def _poll_and_display_streaming_output(self):
        """
        Poll the rank outputs and display any new output in organized sections.
        """
        if self._rank_outputs is None or self._last_displayed_counts is None:
            return
            
        # Check each rank for new output
        for rank in sorted(self._rank_outputs.keys()):
            rank_output = self._rank_outputs[rank]
            last_count = self._last_displayed_counts.get(rank, 0)
            
            # If there's new output for this rank
            if len(rank_output) > last_count:
                new_lines = rank_output[last_count:]
                
                # Display new output in a clean, grouped format
                if last_count == 0:
                    # First output from this rank - show header
                    print(f"\n🔹 Rank {rank}:")
                
                # Display the new lines without rank prefixes (cleaner)
                for line in new_lines:
                    print(f"  {line}")
                
                # Update the count of displayed lines for this rank
                self._last_displayed_counts[rank] = len(rank_output)

    def _enable_distributed_mode(self):
        """
        Enable automatic distributed execution using input transformer.
        
        When enabled, all regular notebook cells are automatically executed across
        all worker processes. Magic commands still execute locally.
        
        The function:
        1. Adds an input transformer to prepend %%distributed to regular cells
        2. Updates the distributed mode state
        
        Note: Timeline tracking is registered separately in __init__ and is always active.
        """
        if not self._distributed_mode_active:
            if self.shell is not None and hasattr(self.shell, 'input_transformers_cleanup'):
                self.shell.input_transformers_cleanup.append(self._distributed_transformer)
                
            self._distributed_mode_active = True

    def _disable_distributed_mode(self):
        """
        Disable automatic distributed execution.
        
        Reverts the notebook to normal local execution by:
        1. Removing the distributed input transformer
        2. Updating the distributed mode state
        
        Note: Timeline tracking handlers remain active and are not unregistered.
        """
        if self._distributed_mode_active:
            if self.shell is not None and hasattr(self.shell, 'input_transformers_cleanup'):
                try:
                    self.shell.input_transformers_cleanup.remove(self._distributed_transformer)
                except ValueError:
                    pass  # Already removed
                    
            self._distributed_mode_active = False

    def _pre_execution_timeline(self, info):
        """
        Pre-execution handler to start timeline tracking for all cells.
        
        Args:
            info: IPython execution info object
        """
        try:
            if hasattr(info, 'raw_cell') and info.raw_cell:
                cell_content = info.raw_cell.strip()
                
                # Don't use execution_count from pre_run_cell as it might not be final
                # We'll update with the correct execution count in post_execution
                cell_id = self._start_cell_execution(cell_content, execution_count=None)
                # Store cell_id in the info object so we can use it in post_execution
                info._timeline_cell_id = cell_id
                
        except Exception:
            pass  # Silently ignore timeline errors

    def _post_execution_sync(self, result):
        """
        Post-execution handler to sync namespaces for IDE support and finish timeline tracking.
        
        After a cell executes in distributed mode, this handler:
        1. Checks if the cell was transformed for distributed execution
        2. If so, syncs the namespace from workers to the local kernel
        3. Enables IDE features like autocomplete for distributed variables
        4. Finishes timeline tracking for regular cells
        
        Args:
            result: IPython execution result object
        """
        try:
            # Handle timeline tracking for regular cells
            if (hasattr(result, 'info') and 
                hasattr(result.info, '_timeline_cell_id')):
                cell_id = result.info._timeline_cell_id
                
                # Update execution count with the actual execution count from the result
                if hasattr(result, 'execution_count') and result.execution_count is not None:
                    updated_cell_id = self._update_execution_count(cell_id, result.execution_count)
                    cell_id = updated_cell_id or cell_id  # Use updated cell_id if available
                
                # Record the execution output and finish timeline
                if hasattr(result, 'result') and result.result is not None:
                    # Record the final result
                    self._record_local_execution_result(cell_id, str(result.result))
                
                # End timeline tracking
                self._end_cell_execution(cell_id)
            
            # Only sync if the cell was transformed for distributed execution
            if (hasattr(result, 'info') and 
                hasattr(result.info, 'raw_cell') and 
                result.info.raw_cell and
                result.info.raw_cell.strip().startswith('%%distributed')):
                if self._comm_manager is not None:
                    self._sync_namespace_to_local()
        except Exception:
            pass  # Silently ignore sync errors

    def _distributed_transformer(self, lines):
        """
        Transform non-magic cells to use %%distributed automatically.
        
        This transformer:
        1. Checks if the cell should be distributed
        2. Prepends %%distributed to eligible cells
        3. Preserves magic commands and comments as local execution
        
        Args:
            lines (List[str]): Lines of code from the notebook cell
            
        Returns:
            List[str]: Transformed lines with %%distributed prepended if appropriate
        """
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
        """
        Show detailed status of distributed workers.
        
        Displays information about each worker process including:
        - Process status (running/stopped)
        - GPU assignment and name
        - Memory usage (if available)
        - Process ID and exit code (if stopped)
        
        Example:
            >>> %dist_status
            Distributed cluster status (4 processes):
            ============================================================
            Rank 0: ✓ PID 12345
              ├─ GPU: 0 (NVIDIA A100)
              ├─ Memory: 10.5GB / 40.0GB (26.2% used)
              └─ Status: Running
        """
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
        """
        Shutdown distributed workers using nuclear option.
        
        This command:
        1. Forces termination of all worker processes
        2. Cleans up communication channels
        3. Disables distributed mode
        4. Resets all internal state
        
        This is a "nuclear" option that ensures all processes are terminated,
        even if they're not responding to normal shutdown signals.
        """
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
        """
        Force shutdown all distributed components without waiting for responses.
        
        This class method:
        1. Attempts graceful shutdown of communication
        2. Forces process termination if graceful shutdown fails
        3. Cleans up all class-level state
        
        This is used as a last resort when normal shutdown fails.
        """
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
        """
        Nuclear option: kill ALL processes related to distributed workers.
        
        This method uses aggressive process termination:
        1. Kills processes by pattern matching
        2. Kills process groups
        3. Forces subprocess status updates
        4. Performs final cleanup of any remaining processes
        
        This ensures no distributed processes remain running, even
        if they're not properly tracked or responding.
        """
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
        """
        Complete reset of distributed environment.
        
        This command:
        1. Performs nuclear shutdown of all processes
        2. Disables distributed mode
        3. Clears all internal state
        4. Prepares the environment for a fresh start
        
        Args:
            line (str): Command line arguments:
                --nuclear/-n: Use nuclear shutdown option
        """
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
        """
        Shutdown all distributed components with graceful cleanup.
        
        This class method:
        1. Sends shutdown signal to workers
        2. Closes communication channels
        3. Terminates worker processes
        4. Cleans up class-level state
        
        This is the preferred shutdown method when processes are responsive.
        """
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
        """
        Execute code on all ranks in the distributed environment.
        
        This magic:
        1. Sends the code to all worker processes
        2. Collects and displays results from each rank
        3. Syncs the namespace back to the local kernel
        4. Records execution timeline for replay
        
        Args:
            line (str): Line magic arguments (unused)
            cell (str): Code to execute on all ranks
            
        Example:
            >>> %%distributed
            >>> import torch
            >>> print(f"Rank {rank}: {torch.cuda.get_device_name()}")
        """
        if not self._comm_manager:
            print("No distributed workers running. Use %dist_init first.")
            return

        # Start timeline tracking
        cell_id = self._start_cell_execution(cell)
        
        try:
            # Initialize streaming buffers for organized output
            self._rank_outputs = {}
            self._last_displayed_counts = {}
            
            # Send to all workers with real-time streaming
            import threading
            import time
            
            # Start execution in a separate thread so we can poll for streaming output
            response_container = {}
            exception_container = {}
            
            def execute_async():
                try:
                    response_container['responses'] = self._comm_manager.send_to_all("execute", cell)
                except Exception as e:
                    exception_container['error'] = e
            
            exec_thread = threading.Thread(target=execute_async)
            exec_thread.start()
            
            # Poll for streaming output while execution is running
            while exec_thread.is_alive():
                self._poll_and_display_streaming_output()
                time.sleep(0.1)  # Poll every 100ms
            
            # Final poll after execution completes
            self._poll_and_display_streaming_output()
            
            # Wait for thread to complete and get results
            exec_thread.join()
            
            if 'error' in exception_container:
                raise exception_container['error']
                
            responses = response_container['responses']
            
            # Record timeline events from responses
            self._record_execution_events(cell_id, responses)
            
            # Only show errors if any occurred (stdout was already streamed)
            for rank, response in responses.items():
                if "error" in response:
                    print(f"\n❌ Error on Rank {rank}: {response['error']}")
                    if "traceback" in response:
                        print(response["traceback"])
            
            # Sync namespace information back to local kernel for IDE support
            self._sync_namespace_to_local()
            
        except Exception as e:
            # Record error event
            self._record_error_event(cell_id, str(e))
            print(f"Error executing distributed code: {e}")
        finally:
            # Clear the streaming buffers
            self._rank_outputs = None
            self._last_displayed_counts = None
            # End timeline tracking
            self._end_cell_execution(cell_id)

    def _sync_namespace_to_local(self):
        """
        Sync variable type information from workers to local IPython kernel.
        
        This method:
        1. Gets namespace information from rank 0
        2. Creates local proxy objects for IDE support
        3. Enables features like autocomplete for distributed variables
        
        The sync focuses on type information to support IDE features
        without copying large data objects.
        """
        try:
            # Get namespace info from rank 0 (representative)
            if self._comm_manager is not None and hasattr(self._comm_manager, 'send_to_ranks'):
                response = self._comm_manager.send_to_ranks([0], "get_namespace_info", "")
                
                if 0 in response and "namespace_info" in response[0]:
                    namespace_info = response[0]["namespace_info"]
                    # Only attempt proxy creation if we have actual namespace data
                    if namespace_info:
                        self._create_local_proxies(namespace_info)
                
        except Exception as e:
            # Don't fail the main execution, just log the sync issue
            print(f"Warning: Could not sync namespace for IDE support: {e}")

    def _create_local_proxies(self, namespace_info: Dict[str, Any]):
        """
        Create proxy objects in local IPython namespace for IDE integration.
        
        This method creates appropriate proxy objects based on type:
        - PyTorch tensors: Creates zero tensors with matching shape/dtype
        - Modules: Imports or creates placeholder modules
        - Functions: Creates stub functions with matching signatures
        - Basic types: Creates matching Python objects
        
        Args:
            namespace_info (Dict[str, Any]): Type information for variables
        """
        try:
            import torch
            from types import ModuleType
            
            # Get the IPython shell's user namespace
            if self.shell is None or not hasattr(self.shell, 'user_ns'):
                return
            user_ns = self.shell.user_ns
            
            for var_name, info in namespace_info.items():
                # Don't skip built-in distributed variables - they're useful for IDE support
                # if var_name in ['rank', 'world_size', 'gpu_id', 'device']:
                #     continue
                    
                # Create appropriate proxy based on type
                if info.get("tensor_type") == "torch.Tensor":
                    # Create a tensor proxy with the right shape and dtype
                    try:
                        shape = info.get("shape", [1])
                        dtype_str = info.get("dtype", "torch.float32")
                        dtype = getattr(torch, dtype_str.split('.')[-1], torch.float32)
                        
                        # Create a small proxy tensor on CPU
                        proxy_tensor = torch.zeros(shape, dtype=dtype)
                        user_ns[var_name] = proxy_tensor
                        
                    except Exception:
                        # Fallback to generic tensor
                        user_ns[var_name] = torch.tensor([0.0])
                        
                elif info.get("torch_device"):
                    # Create a torch.device proxy
                    try:
                        device_type = info.get("device_type", "cpu")
                        device_index = info.get("device_index")
                        if device_index is not None:
                            user_ns[var_name] = torch.device(f"{device_type}:{device_index}")
                        else:
                            user_ns[var_name] = torch.device(device_type)
                    except Exception:
                        # Fallback to CPU device
                        user_ns[var_name] = torch.device("cpu")
                        
                elif info.get("is_module"):
                    # For modules, try to import them locally
                    module_name = info.get("module_name")
                    if module_name:
                        try:
                            # Handle nested module imports properly
                            if '.' in module_name:
                                # For modules like torch.distributed, import the root and navigate
                                root_module = module_name.split('.')[0]
                                # Import safely with a local namespace that includes globals
                                safe_ns = dict(user_ns)
                                safe_ns.update(globals())
                                exec(f"import {root_module}", safe_ns)
                                user_ns[root_module] = safe_ns[root_module]
                                
                                # Navigate to the nested module
                                current_obj = user_ns[root_module]
                                for part in module_name.split('.')[1:]:
                                    current_obj = getattr(current_obj, part)
                                
                                # Assign to the variable name
                                user_ns[var_name] = current_obj
                            else:
                                # Simple import
                                safe_ns = dict(user_ns)
                                safe_ns.update(globals())
                                exec(f"import {module_name}", safe_ns)
                                user_ns[module_name] = safe_ns[module_name]
                                if var_name != module_name:
                                    user_ns[var_name] = user_ns[module_name]
                                    
                        except (ImportError, AttributeError) as e:
                            # Create a placeholder module if import fails
                            placeholder = ModuleType(var_name)
                            placeholder.__file__ = f"<distributed_proxy:{module_name}>"
                            placeholder.__doc__ = f"Proxy module for {module_name} (import failed: {e})"
                            user_ns[var_name] = placeholder
                            
                elif info.get("callable"):
                    # For functions, create a stub with the right signature
                    signature = info.get("signature", "()")
                    doc = info.get("doc", "")
                    
                    # Clean up the signature for valid Python syntax
                    if signature and not signature.startswith('('):
                        signature = f"({signature})"
                    
                    # Create a simple stub function
                    func_code = f"""
def {var_name}{signature}:
    '''
    {doc}
    
    Note: This is a proxy function for IDE support.
    Actual execution happens on distributed workers.
    '''
    raise RuntimeError("This is a proxy function for IDE support only")
"""
                    try:
                        # Use safe namespace for exec
                        safe_ns = dict(user_ns)
                        safe_ns.update(globals())
                        exec(func_code, safe_ns)
                        # Only copy the function we created back to user_ns
                        if var_name in safe_ns:
                            user_ns[var_name] = safe_ns[var_name]
                    except SyntaxError:
                        # Fallback for complex signatures
                        safe_ns = dict(user_ns)
                        safe_ns.update(globals())
                        exec(f"def {var_name}(*args, **kwargs): pass", safe_ns)
                        if var_name in safe_ns:
                            user_ns[var_name] = safe_ns[var_name]
                        
                else:
                    # For other types, create a simple placeholder
                    type_name = info.get("type", "object")
                    class_name = info.get("class_name")
                    
                    if type_name in ["int", "float", "str", "bool", "list", "dict", "tuple"]:
                        # Create basic type instances
                        defaults = {
                            "int": 0, "float": 0.0, "str": "", "bool": False,
                            "list": [], "dict": {}, "tuple": ()
                        }
                        user_ns[var_name] = defaults.get(type_name, None)
                    elif class_name:
                        # Create a placeholder with the right class name (for type hints)
                        try:
                            user_ns[var_name] = type(class_name, (), {})()
                        except:
                            user_ns[var_name] = None
                    else:
                        # Generic placeholder
                        user_ns[var_name] = None
                        
        except Exception as e:
            # Silently skip proxy creation if it fails due to typing issues
            # This commonly happens when cells have no output and typing internals are not accessible
            if "_T_co" not in str(e):
                print(f"Warning: Error creating local proxies: {e}")

    def _record_execution_events(self, cell_id: str, responses: Dict[int, Any]):
        """
        Record execution events from worker responses with line-by-line timing.
        
        Args:
            cell_id (str): The cell execution ID
            responses (Dict[int, Any]): Responses from workers
        """
        if self._timeline_lock is None or self._execution_timelines is None:
            return
            
        with self._timeline_lock:
            if cell_id not in self._execution_timelines:
                return
                
            execution = self._execution_timelines[cell_id]
            
            # Calculate line durations based on cell content
            if execution.cell_content and execution.line_durations is not None:
                cell_lines = execution.cell_content.strip().split('\n')
                line_count = len([line for line in cell_lines if line.strip()])  # Non-empty lines
                
                # Estimate duration per line (we'll refine this with actual execution data)
                if execution.total_duration and line_count > 0:
                    avg_line_duration = execution.total_duration / line_count
                    execution.line_durations = [avg_line_duration] * line_count
            
            for rank, response in responses.items():
                base_timestamp = time.time()
                
                if "error" in response:
                    # Record error event with sub-duration
                    error_duration = 0.001  # Small duration for error events
                    error_event = TimelineEvent(
                        timestamp=base_timestamp,
                        event_type="error",
                        content=response["error"],
                        rank=rank,
                        sub_duration=error_duration
                    )
                    if execution.events is not None:
                        execution.events.append(error_event)
                    
                    if "traceback" in response:
                        traceback_duration = 0.002  # Slightly longer for traceback
                        traceback_event = TimelineEvent(
                            timestamp=base_timestamp + error_duration,
                            event_type="error",
                            content=response["traceback"],
                            rank=rank,
                            sub_duration=traceback_duration
                        )
                        if execution.events is not None:
                            execution.events.append(traceback_event)
                else:
                    # Record output events with calculated line durations
                    if response.get("output"):
                        output_lines = response["output"].strip().split('\n')
                        cumulative_time = 0.0
                        
                        for i, line in enumerate(output_lines):
                            if line.strip():  # Skip empty lines
                                # Calculate sub-duration for this line
                                line_duration = self._calculate_line_duration(line, execution.line_durations, i)
                                
                                output_event = TimelineEvent(
                                    timestamp=base_timestamp + cumulative_time,
                                    event_type="output",
                                    content=line,
                                    rank=rank,
                                    line_number=i + 1,
                                    sub_duration=line_duration
                                )
                                if execution.events is not None:
                                    execution.events.append(output_event)
                                
                                cumulative_time += line_duration

    def _calculate_line_duration(self, line: str, line_durations: Optional[List[float]], line_index: int) -> float:
        """
        Calculate the duration for a specific line based on content and context.
        
        Args:
            line (str): The line content
            line_durations (Optional[List[float]]): Pre-calculated line durations
            line_index (int): Index of the line
            
        Returns:
            float: Duration in seconds for this line
        """
        # If we have pre-calculated durations, use them
        if line_durations and line_index < len(line_durations):
            return line_durations[line_index]
        
        # Otherwise, estimate based on content
        base_duration = 0.001  # 1ms base
        
        # Adjust based on line content
        if any(keyword in line.lower() for keyword in ['print', 'display']):
            return base_duration * 2  # Print statements take a bit longer
        elif any(keyword in line.lower() for keyword in ['import', 'from']):
            return base_duration * 5  # Imports take longer
        elif any(keyword in line.lower() for keyword in ['torch', 'cuda', 'gpu']):
            return base_duration * 3  # GPU operations take longer
        elif len(line.strip()) > 50:
            return base_duration * 2  # Longer lines take more time
        else:
            return base_duration

    def _record_error_event(self, cell_id: str, error_message: str):
        """
        Record an error event in the timeline.
        
        Args:
            cell_id (str): The cell execution ID
            error_message (str): The error message
        """
        if self._timeline_lock is None or self._execution_timelines is None:
            return
            
        with self._timeline_lock:
            if cell_id not in self._execution_timelines:
                return
                
            execution = self._execution_timelines[cell_id]
            error_event = TimelineEvent(
                timestamp=time.time(),
                event_type="error",
                content=error_message,
                sub_duration=0.001  # Small duration for error events
            )
            if execution.events is not None:
                execution.events.append(error_event)

    def _record_local_execution_result(self, cell_id: str, result_content: str):
        """
        Record the result of a local cell execution.
        
        Args:
            cell_id (str): The cell execution ID
            result_content (str): The result content
        """
        if self._timeline_lock is None or self._execution_timelines is None:
            return
            
        with self._timeline_lock:
            if cell_id not in self._execution_timelines:
                return
                
            execution = self._execution_timelines[cell_id]
            result_event = TimelineEvent(
                timestamp=time.time(),
                event_type="output",
                content=result_content,
                line_number=1,
                sub_duration=self._calculate_line_duration(result_content, execution.line_durations, 0)
            )
            if execution.events is not None:
                execution.events.append(result_event)

    @cell_magic
    def rank(self, line, cell):
        """
        Execute code on specific ranks in the distributed environment.
        
        This magic allows targeting specific worker processes:
        1. Parses rank specification (e.g., [0,1] or [0-2])
        2. Sends code only to specified ranks
        3. Displays results from those ranks
        4. Records execution timeline for replay
        
        Args:
            line (str): Rank specification (e.g., "[0,1,2]" or "[0-2]")
            cell (str): Code to execute on specified ranks
            
        Example:
            >>> %%rank[0,1]
            >>> print(f"Running on rank {rank}")
        """
        if not self._comm_manager:
            print("No distributed workers running. Use %dist_init first.")
            return

        # Parse rank specification
        ranks = self._parse_ranks(line)
        if not ranks:
            print("Invalid rank specification. Use: %%rank[0,1,2] or %%rank[0-2]")
            return

        # Start timeline tracking
        cell_id = self._start_cell_execution(cell)
        
        try:
            # Initialize streaming buffers for organized output
            self._rank_outputs = {}
            self._last_displayed_counts = {}
            
            # Send to specified ranks with real-time streaming
            import threading
            import time
            
            # Start execution in a separate thread so we can poll for streaming output
            response_container = {}
            exception_container = {}
            
            def execute_async():
                try:
                    response_container['responses'] = self._comm_manager.send_to_ranks(ranks, "execute", cell)
                except Exception as e:
                    exception_container['error'] = e
            
            exec_thread = threading.Thread(target=execute_async)
            exec_thread.start()
            
            # Poll for streaming output while execution is running
            while exec_thread.is_alive():
                self._poll_and_display_streaming_output()
                time.sleep(0.1)  # Poll every 100ms
            
            # Final poll after execution completes
            self._poll_and_display_streaming_output()
            
            # Wait for thread to complete and get results
            exec_thread.join()
            
            if 'error' in exception_container:
                raise exception_container['error']
                
            responses = response_container['responses']
            
            # Record timeline events from responses
            self._record_execution_events(cell_id, responses)
            
            # Only show errors if any occurred (stdout was already streamed)
            for rank, response in responses.items():
                if "error" in response:
                    print(f"\n❌ Error on Rank {rank}: {response['error']}")
                    if "traceback" in response:
                        print(response["traceback"])
            
        except Exception as e:
            # Record error event
            self._record_error_event(cell_id, str(e))
            print(f"Error executing code on ranks {ranks}: {e}")
        finally:
            # Clear the streaming buffers
            self._rank_outputs = None
            self._last_displayed_counts = None
            # End timeline tracking
            self._end_cell_execution(cell_id)

    @line_magic
    def sync(self, line):
        """
        Synchronize all ranks in the distributed environment.
        
        This magic:
        1. Sends a sync signal to all workers
        2. Waits for acknowledgment from each rank
        3. Ensures all ranks are at the same execution point
        
        This is useful when coordination between ranks is needed.
        """
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
        """
        Display debug information about the distributed state.
        
        Shows detailed information about:
        - Process manager status
        - Communication manager status
        - Number of processes
        - Distributed mode state
        - Individual process status and PIDs
        
        This is useful for diagnosing issues with the distributed setup.
        """
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
        """
        Toggle distributed mode on/off without affecting workers.
        
        This magic controls automatic distributed execution:
        - When enabled: Regular cells execute on all workers
        - When disabled: Regular cells execute locally
        
        Args:
            line (str): Command line arguments:
                --enable/-e: Enable distributed mode
                --disable/-d: Disable distributed mode
                
        Example:
            >>> %dist_mode --enable  # Enable distributed mode
            >>> %dist_mode --disable  # Disable distributed mode
        """
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
        """
        Parse rank specification from string format.
        
        Supports two formats:
        1. Comma-separated list: [0,1,2]
        2. Range specification: [0-2]
        
        Args:
            line (str): Rank specification string
            
        Returns:
            List[int]: List of valid rank numbers
            
        Example:
            >>> _parse_ranks("[0,1,2]")  # Returns [0,1,2]
            >>> _parse_ranks("[0-2]")    # Returns [0,1,2]
        """
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
        """
        Display responses from workers with enhanced REPL-like formatting.
        
        This method:
        1. Formats responses from each rank
        2. Displays output and errors appropriately
        3. Provides visual separation between ranks
        
        Args:
            responses (Dict[int, Any]): Responses from workers
            title (str): Title for the response block
            
        Example output:
            === All ranks ===
            
            --- Rank 0 ---
            Hello from rank 0
            
            --- Rank 1 ---
            Hello from rank 1
        """
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

    @line_magic
    def dist_sync_ide(self, line):
        """
        Manually sync worker namespaces to local kernel for IDE support.
        
        This magic:
        1. Retrieves namespace information from workers
        2. Creates local proxy objects
        3. Enables IDE features like autocomplete
        
        This is useful when automatic sync fails or manual refresh is needed.
        """
        if not self._comm_manager:
            print("No distributed workers running. Use %dist_init first.")
            return
            
        try:
            self._sync_namespace_to_local()
            print("✓ IDE namespace sync completed")
        except Exception as e:
            print(f"❌ Error syncing namespace: {e}")

    @line_magic
    def timeline_save(self, line):
        """
        Manually save timeline data to notebook metadata.
        
        This magic:
        1. Forces the timeline data to be saved to notebook metadata
        2. Displays the current timeline data for verification
        
        Example:
            >>> %timeline_save
        """
        try:
            # Force save the timeline metadata
            self._save_timeline_to_metadata_hook()
            
            # Show current timeline data
            if self._execution_timelines:
                print(f"📊 Timeline data for {len(self._execution_timelines)} cell executions:")
                for cell_id, execution in self._execution_timelines.items():
                    print(f"  • {cell_id}: {execution.total_duration:.4f}s ({len(execution.events or [])} events)")
                print("\n✅ Timeline metadata saved to notebook")
            else:
                print("📊 No timeline data available yet")
                
            # Also show what's in user_ns for debugging
            if (self.shell is not None and 
                hasattr(self.shell, 'user_ns') and 
                '_notebook_execution_timelines' in self.shell.user_ns):
                timeline_data = self.shell.user_ns['_notebook_execution_timelines']
                print(f"🔍 Found {len(timeline_data)} timeline entries in user namespace")
            
        except Exception as e:
            print(f"❌ Error saving timeline: {e}")

    @line_magic  
    def timeline_debug(self, line):
        """
        Show debug information about timeline tracking.
        
        Example:
            >>> %timeline_debug
        """
        print("=== Timeline Debug Information ===")
        print(f"Timeline tracking active: {self._execution_timelines is not None}")
        if self._execution_timelines:
            print(f"Tracked executions: {len(self._execution_timelines)}")
            for cell_id, execution in self._execution_timelines.items():
                print(f"  {cell_id}:")
                print(f"    Duration: {execution.total_duration or 'N/A'}")
                print(f"    Events: {len(execution.events or [])}")
                if execution.line_durations:
                    durations_str = ", ".join(f"{d:.6f}s" for d in execution.line_durations[:5])
                    if len(execution.line_durations) > 5:
                        durations_str += f" ... ({len(execution.line_durations)} total)"
                    print(f"    Line durations: [{durations_str}]")
                else:
                    print(f"    Content preview: {execution.cell_content[:50]}...")
        
        # Check user namespace
        if (self.shell is not None and 
            hasattr(self.shell, 'user_ns')):
            
            if '_notebook_execution_timelines' in self.shell.user_ns:
                saved_data = self.shell.user_ns['_notebook_execution_timelines']
                print(f"\nSaved timeline data: {len(saved_data)} entries")
            else:
                print("\nNo saved timeline data in user namespace")
                
        print("=====================================")

    @line_magic
    def timeline_clear(self, line):
        """
        Clear all timeline data.
        
        Example:
            >>> %timeline_clear
        """
        if self._execution_timelines:
            count = len(self._execution_timelines)
            self._execution_timelines.clear()
            print(f"🗑️ Cleared {count} timeline entries")
        else:
            print("📊 No timeline data to clear")
            
        # Also clear from user namespace
        if (self.shell is not None and 
            hasattr(self.shell, 'user_ns')):
            for key in ['_notebook_execution_timelines', '_timeline_metadata_for_save']:
                if key in self.shell.user_ns:
                    del self.shell.user_ns[key]
                    print(f"🗑️ Cleared {key} from user namespace")


