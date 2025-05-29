"""
Tests for magic.py - IPython magic commands
"""

from unittest.mock import Mock, patch

from jupyter_distributed.magic import DistributedMagic


class TestDistributedMagic:
    """Test magic command functionality"""

    def setup_method(self):
        """Set up test environment"""
        # Reset class variables before each test
        DistributedMagic._process_manager = None
        DistributedMagic._comm_manager = None
        DistributedMagic._num_processes = 0

    def teardown_method(self):
        """Clean up after each test"""
        DistributedMagic.shutdown_all()

    def test_magic_class_initialization(self):
        """Test magic class can be instantiated"""
        magic = DistributedMagic()
        assert magic is not None
        assert DistributedMagic._process_manager is None
        assert DistributedMagic._comm_manager is None
        assert DistributedMagic._num_processes == 0

    @patch("jupyter_distributed.magic.ProcessManager")
    @patch("jupyter_distributed.magic.CommunicationManager")
    def test_dist_init_success(self, mock_comm_manager, mock_process_manager):
        """Test successful distributed initialization"""
        # Mock successful initialization
        mock_pm_instance = Mock()
        mock_pm_instance.start_workers.return_value = 12346
        mock_pm_instance.is_running.return_value = True
        mock_process_manager.return_value = mock_pm_instance

        mock_cm_instance = Mock()
        mock_comm_manager.return_value = mock_cm_instance

        magic = DistributedMagic()

        # Test initialization
        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-ranks 4")

        # Check process manager was called correctly
        mock_pm_instance.start_workers.assert_called_once_with(4, "localhost")

        # Check communication manager was created
        mock_comm_manager.assert_called_once_with(4, 12346)

        # Check state was updated
        assert DistributedMagic._num_processes == 4
        assert DistributedMagic._process_manager == mock_pm_instance
        assert DistributedMagic._comm_manager == mock_cm_instance

        # Check success message was printed
        mock_print.assert_any_call("✓ Successfully started 4 workers")

    @patch("jupyter_distributed.magic.ProcessManager")
    def test_dist_init_failure(self, mock_process_manager):
        """Test distributed initialization failure"""
        # Mock failed initialization
        mock_pm_instance = Mock()
        mock_pm_instance.start_workers.side_effect = Exception("Failed to start")
        mock_process_manager.return_value = mock_pm_instance

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-ranks 2")

        # Check error message was printed
        mock_print.assert_any_call(
            "Failed to start distributed workers: Failed to start"
        )

        # Check state wasn't updated
        assert DistributedMagic._num_processes == 0

    def test_dist_init_already_running(self):
        """Test dist_init when workers already running"""
        # Mock already running process manager
        mock_pm = Mock()
        mock_pm.is_running.return_value = True
        DistributedMagic._process_manager = mock_pm

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-ranks 2")

        mock_print.assert_called_with(
            "Distributed workers already running. Use %dist_shutdown to stop them first."
        )

    def test_dist_status_no_workers(self):
        """Test status when no workers are running"""
        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_status("")

        mock_print.assert_called_with("No distributed workers running")

    def test_dist_status_with_workers(self):
        """Test status with running workers"""
        # Mock process manager with status
        mock_pm = Mock()
        mock_pm.get_status.return_value = {
            0: {"pid": 12345, "running": True, "returncode": None},
            1: {"pid": 12346, "running": False, "returncode": 1},
        }
        DistributedMagic._process_manager = mock_pm
        DistributedMagic._num_processes = 2

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_status("")

        # Check status output
        mock_print.assert_any_call("Distributed cluster status (2 ranks):")
        mock_print.assert_any_call("  Rank 0: ✓ PID 12345")
        mock_print.assert_any_call("  Rank 1: ✗ PID 12346")

    def test_dist_shutdown(self):
        """Test distributed shutdown"""
        magic = DistributedMagic()

        with patch.object(DistributedMagic, "shutdown_all") as mock_shutdown:
            with patch("builtins.print") as mock_print:
                magic.dist_shutdown("")

            mock_shutdown.assert_called_once()
            mock_print.assert_called_with("Distributed workers shutdown")

    def test_shutdown_all(self):
        """Test shutdown_all class method"""
        # Mock managers
        mock_cm = Mock()
        mock_pm = Mock()

        DistributedMagic._comm_manager = mock_cm
        DistributedMagic._process_manager = mock_pm
        DistributedMagic._num_processes = 2

        DistributedMagic.shutdown_all()

        # Check shutdown was called
        mock_cm.send_to_all.assert_called_once_with("shutdown", {})
        mock_cm.shutdown.assert_called_once()
        mock_pm.shutdown.assert_called_once()

        # Check state was reset
        assert DistributedMagic._comm_manager is None
        assert DistributedMagic._process_manager is None

    def test_shutdown_all_with_exception(self):
        """Test shutdown_all handles exceptions gracefully"""
        # Mock communication manager that raises exception
        mock_cm = Mock()
        mock_cm.send_to_all.side_effect = Exception("Communication error")
        mock_pm = Mock()

        DistributedMagic._comm_manager = mock_cm
        DistributedMagic._process_manager = mock_pm

        # Should not raise exception
        DistributedMagic.shutdown_all()

        # Should still clean up
        assert DistributedMagic._comm_manager is None
        assert DistributedMagic._process_manager is None

    def test_distributed_cell_no_workers(self):
        """Test %%distributed magic with no workers"""
        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.distributed("", "print('hello')")

        mock_print.assert_called_with(
            "No distributed workers running. Use %dist_init first."
        )

    def test_distributed_cell_success(self):
        """Test %%distributed magic with successful execution"""
        # Mock communication manager
        mock_cm = Mock()
        mock_cm.send_to_all.return_value = {
            0: {"output": "hello from rank 0", "status": "success"},
            1: {"output": "hello from rank 1", "status": "success"},
        }
        DistributedMagic._comm_manager = mock_cm

        magic = DistributedMagic()

        with patch.object(magic, "_display_responses") as mock_display:
            magic.distributed("", "print('hello')")

        mock_cm.send_to_all.assert_called_once_with("execute", "print('hello')")
        mock_display.assert_called_once()

    def test_distributed_cell_error(self):
        """Test %%distributed magic with execution error"""
        # Mock communication manager that raises exception
        mock_cm = Mock()
        mock_cm.send_to_all.side_effect = Exception("Execution failed")
        DistributedMagic._comm_manager = mock_cm

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.distributed("", "print('hello')")

        mock_print.assert_called_with(
            "Error executing distributed code: Execution failed"
        )

    def test_rank_cell_no_workers(self):
        """Test %%rank magic with no workers"""
        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.rank("[0,1]", "print('hello')")

        mock_print.assert_called_with(
            "No distributed workers running. Use %dist_init first."
        )

    def test_rank_cell_invalid_spec(self):
        """Test %%rank magic with invalid rank specification"""
        DistributedMagic._comm_manager = Mock()
        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.rank("invalid", "print('hello')")

        mock_print.assert_called_with(
            "Invalid rank specification. Use: %%rank[0,1,2] or %%rank[0-2]"
        )

    def test_rank_cell_success(self):
        """Test %%rank magic with successful execution"""
        # Mock communication manager
        mock_cm = Mock()
        mock_cm.send_to_ranks.return_value = {
            0: {"output": "hello from rank 0", "status": "success"},
            2: {"output": "hello from rank 2", "status": "success"},
        }
        DistributedMagic._comm_manager = mock_cm
        DistributedMagic._num_processes = 4

        magic = DistributedMagic()

        with patch.object(magic, "_parse_ranks", return_value=[0, 2]) as mock_parse:
            with patch.object(magic, "_display_responses") as mock_display:
                magic.rank("[0,2]", "print('hello')")

        mock_cm.send_to_ranks.assert_called_once_with(
            [0, 2], "execute", "print('hello')"
        )
        mock_display.assert_called_once_with(
            {
                0: {"output": "hello from rank 0", "status": "success"},
                2: {"output": "hello from rank 2", "status": "success"},
            },
            "Ranks [0, 2]",
        )

    def test_sync_no_workers(self):
        """Test %sync magic with no workers"""
        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.sync("")

        mock_print.assert_called_with(
            "No distributed workers running. Use %dist_init first."
        )

    def test_sync_success(self):
        """Test %sync magic with successful synchronization"""
        # Mock communication manager
        mock_cm = Mock()
        mock_cm.send_to_all.return_value = {
            0: {"status": "synced"},
            1: {"status": "synced"},
        }
        DistributedMagic._comm_manager = mock_cm

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.sync("")

        mock_cm.send_to_all.assert_called_once_with("sync", {})
        mock_print.assert_called_with("✓ Synchronized 2 ranks")

    def test_parse_ranks_simple(self):
        """Test parsing simple rank specifications"""
        DistributedMagic._num_processes = 4
        magic = DistributedMagic()

        # Test simple list
        ranks = magic._parse_ranks("[0,1,2]")
        assert ranks == [0, 1, 2]

        # Test range
        ranks = magic._parse_ranks("[0-2]")
        assert ranks == [0, 1, 2]

        # Test mixed
        ranks = magic._parse_ranks("[0,2-3]")
        assert ranks == [0, 2, 3]

    def test_parse_ranks_filtering(self):
        """Test that parse_ranks filters invalid ranks"""
        DistributedMagic._num_processes = 2
        magic = DistributedMagic()

        # Should filter out rank 3 (out of range)
        ranks = magic._parse_ranks("[0,1,3]")
        assert ranks == [0, 1]

    def test_parse_ranks_invalid_format(self):
        """Test parse_ranks with invalid format"""
        magic = DistributedMagic()

        # Invalid format should return empty list
        ranks = magic._parse_ranks("0,1,2")  # Missing brackets
        assert ranks == []

        ranks = magic._parse_ranks("[")  # Incomplete brackets
        assert ranks == []

    def test_display_responses_success(self):
        """Test _display_responses with successful responses"""
        magic = DistributedMagic()

        responses = {
            0: {"output": "Hello from rank 0", "status": "success"},
            1: {"output": "Hello from rank 1", "status": "success"},
        }

        with patch("builtins.print") as mock_print:
            magic._display_responses(responses, "Test ranks")

        # Check output format
        mock_print.assert_any_call("\n=== Test ranks ===")
        mock_print.assert_any_call("\n--- Rank 0 ---")
        mock_print.assert_any_call("Hello from rank 0")
        mock_print.assert_any_call("\n--- Rank 1 ---")
        mock_print.assert_any_call("Hello from rank 1")

    def test_display_responses_error(self):
        """Test _display_responses with error responses"""
        magic = DistributedMagic()

        responses = {
            0: {"error": "Syntax error", "traceback": "Traceback..."},
            1: {"output": "Success", "status": "success"},
        }

        with patch("builtins.print") as mock_print:
            magic._display_responses(responses, "Test ranks")

        # Check error formatting
        mock_print.assert_any_call("❌ Error: Syntax error")
        mock_print.assert_any_call("Traceback...")
        mock_print.assert_any_call("Success")

    def test_display_responses_no_output(self):
        """Test _display_responses with no output"""
        magic = DistributedMagic()

        responses = {
            0: {"status": "success"}  # No output field
        }

        with patch("builtins.print") as mock_print:
            magic._display_responses(responses, "Test ranks")

        mock_print.assert_any_call("✓ Executed successfully")


class TestDistributedMagicIntegration:
    """Integration tests for magic commands"""

    def setup_method(self):
        """Set up test environment"""
        DistributedMagic._process_manager = None
        DistributedMagic._comm_manager = None
        DistributedMagic._num_processes = 0

    def teardown_method(self):
        """Clean up after each test"""
        DistributedMagic.shutdown_all()

    @patch("jupyter_distributed.magic.ProcessManager")
    @patch("jupyter_distributed.magic.CommunicationManager")
    def test_full_workflow(self, mock_comm_manager, mock_process_manager):
        """Test complete workflow: init -> execute -> shutdown"""
        # Mock successful components
        mock_pm = Mock()
        mock_pm.start_workers.return_value = 12346
        mock_pm.is_running.return_value = True
        mock_pm.get_status.return_value = {
            0: {"pid": 123, "running": True, "returncode": None}
        }
        mock_process_manager.return_value = mock_pm

        mock_cm = Mock()
        mock_cm.send_to_all.return_value = {0: {"output": "Hello", "status": "success"}}
        mock_comm_manager.return_value = mock_cm

        magic = DistributedMagic()

        # Test init
        with patch("builtins.print"):
            magic.dist_init("--num-ranks 1")

        assert DistributedMagic._num_processes == 1

        # Test status
        with patch("builtins.print") as mock_print:
            magic.dist_status("")

        mock_print.assert_any_call("Distributed cluster status (1 ranks):")

        # Test execution
        with patch("builtins.print"):
            magic.distributed("", "print('Hello')")

        mock_cm.send_to_all.assert_called_with("execute", "print('Hello')")

        # Test shutdown
        with patch("builtins.print"):
            magic.dist_shutdown("")

        assert DistributedMagic._num_processes == 0
