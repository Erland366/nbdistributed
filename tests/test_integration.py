"""
Integration tests for the complete jupyter_distributed system
"""

import pytest
import time
import tempfile
import os
from unittest.mock import patch, Mock

try:
    import nbformat
    from nbclient import NotebookClient

    NBCLIENT_AVAILABLE = True
except ImportError:
    NBCLIENT_AVAILABLE = False

from jupyter_distributed.magic import DistributedMagic
from jupyter_distributed.process_manager import ProcessManager
from jupyter_distributed.communication import CommunicationManager


class NotebookTestRunner:
    """Custom test runner for Jupyter notebooks"""

    def __init__(self, num_processes=2):
        self.num_processes = num_processes
        self.temp_dir = None
        self.notebook_path = None

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir)

    def create_test_notebook(self, cells):
        """Create a test notebook with given cells"""
        if not NBCLIENT_AVAILABLE:
            pytest.skip("nbformat and nbclient not available")

        nb = nbformat.v4.new_notebook()

        for cell_content in cells:
            if cell_content.startswith("%"):
                # Magic command
                cell = nbformat.v4.new_code_cell(cell_content)
            else:
                # Regular code
                cell = nbformat.v4.new_code_cell(cell_content)
            nb.cells.append(cell)

        self.notebook_path = os.path.join(self.temp_dir, "test_notebook.ipynb")
        with open(self.notebook_path, "w") as f:
            nbformat.write(nb, f)

        return self.notebook_path

    def execute_notebook(self, notebook_path=None):
        """Execute the test notebook"""
        if not NBCLIENT_AVAILABLE:
            pytest.skip("nbformat and nbclient not available")

        if notebook_path is None:
            notebook_path = self.notebook_path

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Create a client and execute
        client = NotebookClient(nb, timeout=60)

        try:
            client.execute()
            return nb, None
        except Exception as e:
            return nb, str(e)


@pytest.mark.integration
class TestFullSystemIntegration:
    """Test the complete system working together"""

    def setup_method(self):
        """Clean up before each test"""
        DistributedMagic.shutdown_all()

    def teardown_method(self):
        """Clean up after each test"""
        DistributedMagic.shutdown_all()

    @pytest.mark.slow
    @patch("subprocess.Popen")
    def test_magic_commands_integration(self, mock_popen):
        """Test magic commands working together"""
        # Mock successful process creation
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        magic = DistributedMagic()

        # Test initialization
        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-processes 2")

        # Verify workers were started
        assert DistributedMagic._num_processes == 2
        assert DistributedMagic._process_manager is not None
        assert DistributedMagic._comm_manager is not None

        # Test status
        with patch("builtins.print") as mock_print:
            magic.dist_status("")

        mock_print.assert_any_call("Distributed cluster status (2 ranks):")

        # Test shutdown
        with patch("builtins.print"):
            magic.dist_shutdown("")

        assert DistributedMagic._num_processes == 0

    @pytest.mark.slow
    def test_process_communication_integration(self):
        """Test process manager and communication working together"""
        process_manager = ProcessManager()

        try:
            # Mock the worker startup to avoid actual PyTorch distributed init
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                comm_port = process_manager.start_workers(num_processes=2)

                # Verify process manager state
                assert process_manager.num_processes == 2
                assert process_manager.is_running()
                assert isinstance(comm_port, int)

                # Test communication manager can be created
                comm_manager = CommunicationManager(num_processes=2, base_port=comm_port)

                # Verify communication manager state
                assert comm_manager.num_processes == 2
                assert comm_manager.running

                comm_manager.shutdown()

        finally:
            process_manager.shutdown()

    @pytest.mark.skipif(not NBCLIENT_AVAILABLE, reason="nbclient not available")
    def test_notebook_execution_mocked(self):
        """Test executing a notebook with mocked distributed workers"""
        with NotebookTestRunner(num_processes=2) as runner:
            # Create test notebook
            cells = [
                "%load_ext jupyter_distributed",
                "%dist_init --num-processes 2",
                "%%distributed\nprint('Hello from all ranks')",
                "%dist_status",
                "%dist_shutdown",
            ]

            notebook_path = runner.create_test_notebook(cells)

            # Mock the distributed components
            with (
                patch("jupyter_distributed.magic.ProcessManager") as mock_pm_class,
                patch(
                    "jupyter_distributed.magic.CommunicationManager"
                ) as mock_cm_class,
            ):
                # Mock successful initialization
                mock_pm = Mock()
                mock_pm.start_workers.return_value = 12346
                mock_pm.is_running.return_value = True
                mock_pm.get_status.return_value = {
                    0: {"pid": 123, "running": True, "returncode": None},
                    1: {"pid": 124, "running": True, "returncode": None},
                }
                mock_pm_class.return_value = mock_pm

                mock_cm = Mock()
                mock_cm.send_to_all.return_value = {
                    0: {"output": "Hello from all ranks\n", "status": "success"},
                    1: {"output": "Hello from all ranks\n", "status": "success"},
                }
                mock_cm_class.return_value = mock_cm

                # Execute notebook
                nb, error = runner.execute_notebook(notebook_path)

                # Check that no errors occurred
                if error:
                    pytest.fail(f"Notebook execution failed: {error}")

                # Verify cells executed successfully
                assert len(nb.cells) == 5
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        assert cell.execution_count is not None


class TestErrorHandling:
    """Test error handling across the system"""

    def setup_method(self):
        DistributedMagic.shutdown_all()

    def teardown_method(self):
        DistributedMagic.shutdown_all()

    def test_worker_startup_failure(self):
        """Test handling of worker startup failures"""
        with patch("subprocess.Popen") as mock_popen:
            # Mock process that dies immediately
            mock_process = Mock()
            mock_process.poll.return_value = 1  # Exited with error
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            with patch("builtins.print") as mock_print:
                magic.dist_init("--num-processes 2")

            # Should show error message
            mock_print.assert_any_call(
                "Failed to start distributed workers: Worker 0 failed to start"
            )

            # Should not have workers running
            assert DistributedMagic._num_processes == 0

    def test_communication_timeout(self):
        """Test handling of communication timeouts"""
        # Mock successful startup but failing communication
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            with patch("builtins.print"):
                magic.dist_init("--num-processes 2")

            # Mock communication timeout
            DistributedMagic._comm_manager.send_to_all = Mock(
                side_effect=TimeoutError("Timeout")
            )

            with patch("builtins.print") as mock_print:
                magic.distributed("", "print('hello')")

            mock_print.assert_called_with("Error executing distributed code: Timeout")

    def test_partial_worker_failure(self):
        """Test handling when some workers fail during execution"""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            with patch("builtins.print"):
                magic.dist_init("--num-processes 2")

            # Mock partial failure response
            DistributedMagic._comm_manager.send_to_all = Mock(
                return_value={
                    0: {"output": "Success", "status": "success"},
                    1: {"error": "Failed", "traceback": "Error trace"},
                }
            )

            with patch("builtins.print") as mock_print:
                magic.distributed("", "print('hello')")

            # Should display both success and error
            mock_print.assert_any_call("Success")
            mock_print.assert_any_call("❌ Error: Failed")


class TestPerformance:
    """Test performance characteristics"""

    def setup_method(self):
        DistributedMagic.shutdown_all()

    def teardown_method(self):
        DistributedMagic.shutdown_all()

    @pytest.mark.slow
    def test_startup_shutdown_time(self):
        """Test that startup and shutdown happen in reasonable time"""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None  # Running
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            # Test startup time
            start_time = time.time()
            with patch("builtins.print"):
                magic.dist_init("--num-processes 4")
            startup_time = time.time() - start_time

            assert startup_time < 5.0  # Should start within 5 seconds
            assert DistributedMagic._num_processes == 4

            # Test shutdown time
            start_time = time.time()
            with patch("builtins.print"):
                magic.dist_shutdown("")
            shutdown_time = time.time() - start_time

            assert shutdown_time < 2.0  # Should shutdown within 2 seconds
            assert DistributedMagic._num_processes == 0

    def test_multiple_rapid_commands(self):
        """Test rapid execution of multiple commands"""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            with patch("builtins.print"):
                magic.dist_init("--num-processes 2")

            # Mock fast communication
            DistributedMagic._comm_manager.send_to_all = Mock(
                return_value={
                    0: {"output": "OK", "status": "success"},
                    1: {"output": "OK", "status": "success"},
                }
            )

            # Execute multiple commands rapidly
            start_time = time.time()
            for i in range(10):
                with patch("builtins.print"):
                    magic.distributed("", f"x = {i}")
            execution_time = time.time() - start_time

            # Should handle rapid commands without issues
            assert execution_time < 1.0
            assert DistributedMagic._comm_manager.send_to_all.call_count == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def setup_method(self):
        DistributedMagic.shutdown_all()

    def teardown_method(self):
        DistributedMagic.shutdown_all()

    def test_single_rank_cluster(self):
        """Test cluster with only one rank"""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            with patch("builtins.print") as mock_print:
                magic.dist_init("--num-processes 1")

            assert DistributedMagic._num_processes == 1
            mock_print.assert_any_call("✓ Successfully started 1 workers")

    def test_large_rank_count(self):
        """Test cluster with many ranks"""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            with patch("builtins.print") as mock_print:
                magic.dist_init("--num-processes 16")

            assert DistributedMagic._num_processes == 16
            mock_print.assert_any_call("✓ Successfully started 16 workers")

    def test_complex_rank_specifications(self):
        """Test complex rank specifications"""
        DistributedMagic._num_processes = 8
        magic = DistributedMagic()

        # Test various complex specifications
        test_cases = [
            ("[0,2,4,6]", [0, 2, 4, 6]),
            ("[0-3,6-7]", [0, 1, 2, 3, 6, 7]),
            ("[1,3-5,7]", [1, 3, 4, 5, 7]),
            ("[0-7]", [0, 1, 2, 3, 4, 5, 6, 7]),
        ]

        for spec, expected in test_cases:
            result = magic._parse_ranks(spec)
            assert result == expected, (
                f"Failed for spec {spec}: got {result}, expected {expected}"
            )

    def test_empty_code_execution(self):
        """Test executing empty or whitespace-only code"""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            magic = DistributedMagic()

            with patch("builtins.print"):
                magic.dist_init("--num-processes 2")

            # Mock communication
            DistributedMagic._comm_manager.send_to_all = Mock(
                return_value={
                    0: {"output": "", "status": "success"},
                    1: {"output": "", "status": "success"},
                }
            )

            # Test empty code
            with patch("builtins.print"):
                magic.distributed("", "")

            # Test whitespace-only code
            with patch("builtins.print"):
                magic.distributed("", "   \n\t  ")

            # Should handle gracefully
            assert DistributedMagic._comm_manager.send_to_all.call_count == 2


# Custom pytest markers
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


if __name__ == "__main__":
    # Allow running integration tests directly
    pytest.main([__file__, "-v", "--tb=short"])
