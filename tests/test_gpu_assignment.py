"""
Tests for GPU assignment functionality
"""

import pytest
from unittest.mock import Mock, patch

from jupyter_distributed.magic import DistributedMagic
from jupyter_distributed.process_manager import ProcessManager


class TestGPUAssignment:
    """Test GPU ID assignment functionality"""

    def setup_method(self):
        """Clean up before each test"""
        DistributedMagic.shutdown_all()

    def teardown_method(self):
        """Clean up after each test"""
        DistributedMagic.shutdown_all()

    @patch("subprocess.Popen")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_gpu_ids_validation_success(
        self, mock_device_count, mock_cuda_available, mock_popen
    ):
        """Test successful GPU ID validation"""
        # Mock CUDA environment
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 4  # 4 GPUs available (0,1,2,3)

        # Mock successful process creation
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-ranks 2 --gpu-ids 0,1")

        # Should succeed and show GPU assignments
        mock_print.assert_any_call("Using GPU IDs: [0, 1]")
        mock_print.assert_any_call("✓ Successfully started 2 workers")
        mock_print.assert_any_call("  Rank 0 -> GPU 0")
        mock_print.assert_any_call("  Rank 1 -> GPU 1")

        assert DistributedMagic._num_ranks == 2

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_gpu_ids_validation_invalid_ids(
        self, mock_device_count, mock_cuda_available
    ):
        """Test validation with invalid GPU IDs"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2  # Only GPUs 0,1 available

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-ranks 2 --gpu-ids 0,3")  # GPU 3 doesn't exist

        # Should fail validation
        mock_print.assert_any_call("❌ Invalid GPU IDs: [3]")
        mock_print.assert_any_call("Available GPUs: [0, 1]")
        assert DistributedMagic._num_ranks == 0

    @patch("torch.cuda.is_available")
    def test_gpu_ids_validation_not_enough_gpus(self, mock_cuda_available):
        """Test validation when not enough GPU IDs provided"""
        mock_cuda_available.return_value = True

        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-ranks 4 --gpu-ids 0,1")  # Need 4 ranks, only 2 GPUs

        # Should fail validation
        mock_print.assert_any_call("❌ Not enough GPU IDs specified. Need 4, got 2")
        assert DistributedMagic._num_ranks == 0

    @patch("torch.cuda.is_available")
    def test_gpu_ids_no_cuda(self, mock_cuda_available):
        """Test behavior when CUDA is not available"""
        mock_cuda_available.return_value = False

        magic = DistributedMagic()

        with (
            patch("builtins.print") as mock_print,
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            magic.dist_init("--num-ranks 2 --gpu-ids 0,1")

        # Should warn and continue without GPU IDs
        mock_print.assert_any_call("⚠️  CUDA not available, GPU IDs will be ignored")
        mock_print.assert_any_call("✓ Successfully started 2 workers")
        assert DistributedMagic._num_ranks == 2

    def test_gpu_ids_invalid_format(self):
        """Test invalid GPU ID format"""
        magic = DistributedMagic()

        with patch("builtins.print") as mock_print:
            magic.dist_init("--num-ranks 2 --gpu-ids abc,def")

        mock_print.assert_called_with(
            "❌ Invalid GPU IDs format. Use comma-separated integers (e.g., '0,1,3')"
        )
        assert DistributedMagic._num_ranks == 0


class TestProcessManagerGPUAssignment:
    """Test ProcessManager GPU assignment"""

    @patch("subprocess.Popen")
    def test_start_workers_with_gpu_ids(self, mock_popen):
        """Test starting workers with specific GPU IDs"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = ProcessManager()

        try:
            manager.start_workers(num_ranks=2, gpu_ids=[0, 3])

            # Check that processes were called with correct GPU IDs
            assert mock_popen.call_count == 2

            # Check first worker (rank 0, GPU 0)
            first_call_args = mock_popen.call_args_list[0][0][0]
            assert first_call_args[-1] == "0"  # GPU ID should be last argument

            # Check second worker (rank 1, GPU 3)
            second_call_args = mock_popen.call_args_list[1][0][0]
            assert second_call_args[-1] == "3"  # GPU ID should be last argument

        finally:
            manager.shutdown()

    @patch("subprocess.Popen")
    def test_start_workers_without_gpu_ids(self, mock_popen):
        """Test starting workers without GPU IDs (auto-assignment)"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = ProcessManager()

        try:
            manager.start_workers(num_ranks=2)

            # Check that processes were called without GPU ID argument
            assert mock_popen.call_count == 2

            # Check that command doesn't include GPU ID
            for call in mock_popen.call_args_list:
                cmd = call[0][0]
                assert len(cmd) == 6  # Should have 6 arguments (without GPU ID)

        finally:
            manager.shutdown()

    @patch("subprocess.Popen")
    def test_gpu_id_cycling(self, mock_popen):
        """Test GPU ID cycling when more ranks than GPU IDs"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = ProcessManager()

        try:
            # 4 ranks, but only 2 GPU IDs - should cycle
            manager.start_workers(num_ranks=4, gpu_ids=[0, 1])

            assert mock_popen.call_count == 4

            # Check GPU ID assignments: 0, 1, 0, 1
            expected_gpu_ids = ["0", "1", "0", "1"]
            for i, call in enumerate(mock_popen.call_args_list):
                cmd = call[0][0]
                actual_gpu_id = cmd[-1]
                assert actual_gpu_id == expected_gpu_ids[i]

        finally:
            manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
