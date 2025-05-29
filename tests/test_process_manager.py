"""
Tests for process_manager.py
"""

import pytest
import subprocess
from unittest.mock import Mock, patch

from jupyter_distributed.process_manager import ProcessManager


class TestProcessManager:
    """Test ProcessManager functionality"""

    def test_initialization(self):
        """Test process manager initialization"""
        manager = ProcessManager()

        assert manager.processes == []
        assert manager.num_ranks == 0
        assert manager.master_port is None
        assert manager.comm_port is None

    def test_find_free_port(self):
        """Test finding free ports"""
        manager = ProcessManager()

        port1 = manager._find_free_port()
        port2 = manager._find_free_port()

        # Should get different ports
        assert port1 != port2
        assert 1024 < port1 < 65535
        assert 1024 < port2 < 65535

    @patch("subprocess.Popen")
    def test_start_workers_success(self, mock_popen):
        """Test successful worker startup"""
        # Mock successful process creation
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process

        manager = ProcessManager()

        with patch.object(manager, "_find_free_port", side_effect=[12345, 12346]):
            comm_port = manager.start_workers(num_ranks=2)

        # Check state
        assert manager.num_ranks == 2
        assert manager.master_port == 12345
        assert manager.comm_port == 12346
        assert comm_port == 12346
        assert len(manager.processes) == 2

        # Check subprocess calls
        assert mock_popen.call_count == 2

        # Verify command arguments
        for call_idx, call in enumerate(mock_popen.call_args_list):
            args, kwargs = call
            cmd = args[0]

            assert "python" in cmd[0]
            assert cmd[1].endswith("worker.py")
            assert cmd[2] == str(call_idx)  # rank
            assert cmd[3] == "2"  # world_size
            assert cmd[4] == "localhost"  # master_addr
            assert cmd[5] == "12345"  # master_port
            assert cmd[6] == "12346"  # comm_port

    @patch("subprocess.Popen")
    def test_start_workers_failure(self, mock_popen):
        """Test worker startup failure"""
        # Mock process that dies immediately
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited with error
        mock_popen.return_value = mock_process

        manager = ProcessManager()

        with patch.object(manager, "_find_free_port", side_effect=[12345, 12346]):
            with pytest.raises(RuntimeError, match="Worker 0 failed to start"):
                manager.start_workers(num_ranks=2)

        # Should have cleaned up
        assert len(manager.processes) == 0
        assert manager.num_ranks == 0

    def test_shutdown_no_processes(self):
        """Test shutdown when no processes are running"""
        manager = ProcessManager()

        # Should not raise any errors
        manager.shutdown()

        assert manager.processes == []
        assert manager.num_ranks == 0

    def test_shutdown_with_processes(self):
        """Test shutdown with running processes"""
        manager = ProcessManager()

        # Create mock processes
        mock_process1 = Mock()
        mock_process1.poll.return_value = None  # Still running
        mock_process1.pid = 12345

        mock_process2 = Mock()
        mock_process2.poll.return_value = None  # Still running
        mock_process2.pid = 12346

        manager.processes = [mock_process1, mock_process2]
        manager.num_ranks = 2

        # Test shutdown
        manager.shutdown()

        # Should have called terminate on both
        mock_process1.terminate.assert_called_once()
        mock_process2.terminate.assert_called_once()

        # Should have waited for both
        mock_process1.wait.assert_called_once_with(timeout=5)
        mock_process2.wait.assert_called_once_with(timeout=5)

        # Should have cleared state
        assert manager.processes == []
        assert manager.num_ranks == 0

    def test_shutdown_with_hanging_processes(self):
        """Test shutdown when processes don't terminate gracefully"""
        manager = ProcessManager()

        # Create mock process that doesn't terminate
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired("test", 5)
        mock_process.pid = 12345

        manager.processes = [mock_process]
        manager.num_ranks = 1

        # Test shutdown
        manager.shutdown()

        # Should have tried terminate first
        mock_process.terminate.assert_called_once()

        # Should have tried graceful wait
        mock_process.wait.assert_called_once_with(timeout=5)

        # Should have killed when timeout
        mock_process.kill.assert_called_once()

        # Should have cleared state
        assert manager.processes == []
        assert manager.num_ranks == 0

    def test_is_running_no_processes(self):
        """Test is_running with no processes"""
        manager = ProcessManager()

        assert manager.is_running() is False

    def test_is_running_with_processes(self):
        """Test is_running with active processes"""
        manager = ProcessManager()

        # Mock running process
        mock_process1 = Mock()
        mock_process1.poll.return_value = None  # Still running

        # Mock dead process
        mock_process2 = Mock()
        mock_process2.poll.return_value = 1  # Exited

        manager.processes = [mock_process1, mock_process2]

        # Should return True if any process is running
        assert manager.is_running() is True

        # If all processes are dead
        mock_process1.poll.return_value = 1
        assert manager.is_running() is False

    def test_get_status(self):
        """Test getting process status"""
        manager = ProcessManager()

        # Create mock processes
        mock_process1 = Mock()
        mock_process1.pid = 12345
        mock_process1.poll.return_value = None  # Running
        mock_process1.returncode = None

        mock_process2 = Mock()
        mock_process2.pid = 12346
        mock_process2.poll.return_value = 1  # Exited
        mock_process2.returncode = 1

        manager.processes = [mock_process1, mock_process2]

        status = manager.get_status()

        expected = {
            0: {"pid": 12345, "running": True, "returncode": None},
            1: {"pid": 12346, "running": False, "returncode": 1},
        }

        assert status == expected


class TestProcessManagerIntegration:
    """Integration tests for ProcessManager"""

    @pytest.mark.slow
    def test_start_and_shutdown_real_workers(self):
        """Test starting and shutting down real worker processes"""
        manager = ProcessManager()

        try:
            # This might fail if torch/distributed not available
            # but we want to test the process management logic
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                comm_port = manager.start_workers(num_ranks=2)

                assert manager.is_running()
                assert manager.num_ranks == 2
                assert isinstance(comm_port, int)

                status = manager.get_status()
                assert len(status) == 2

        finally:
            manager.shutdown()
            assert not manager.is_running()

    def test_multiple_start_shutdown_cycles(self):
        """Test multiple start/shutdown cycles"""
        manager = ProcessManager()

        for i in range(3):
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                manager.start_workers(num_ranks=2)
                assert manager.is_running()

                manager.shutdown()
                assert not manager.is_running()

    def test_port_allocation(self):
        """Test that ports are properly allocated and freed"""
        manager1 = ProcessManager()
        manager2 = ProcessManager()

        try:
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                # Start first manager
                port1 = manager1.start_workers(num_ranks=2)

                # Start second manager (should get different ports)
                port2 = manager2.start_workers(num_ranks=2)

                assert port1 != port2
                assert manager1.master_port != manager2.master_port
                assert manager1.comm_port != manager2.comm_port

        finally:
            manager1.shutdown()
            manager2.shutdown()
