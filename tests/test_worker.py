"""
Tests for worker.py
"""

import pytest
import time
import os
from unittest.mock import Mock, patch
import torch
import zmq
import pickle

from jupyter_distributed.worker import DistributedWorker
from jupyter_distributed.communication import Message


class TestDistributedWorker:
    """Test DistributedWorker functionality"""

    @patch("torch.distributed.init_process_group")
    @patch("zmq.Context")
    def test_initialization(self, mock_zmq_context, mock_init_pg):
        """Test worker initialization"""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context

        worker = DistributedWorker(
            rank=0,
            world_size=2,
            master_addr="localhost",
            master_port="12345",
            comm_port=12346,
        )

        # Check environment variables were set
        assert os.environ["RANK"] == "0"
        assert os.environ["LOCAL_RANK"] == "0"
        assert os.environ["WORLD_SIZE"] == "2"
        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "12345"

        # Check distributed init was called
        mock_init_pg.assert_called_once()

        # Check namespace setup
        assert worker.namespace["rank"] == 0
        assert worker.namespace["world_size"] == 2
        assert "torch" in worker.namespace
        assert "dist" in worker.namespace

        # Check ZMQ setup
        mock_socket.setsockopt.assert_called_with(zmq.IDENTITY, b"worker_0")
        mock_socket.connect.assert_called_with("tcp://localhost:12346")

    @patch("torch.cuda.is_available")
    @patch("torch.distributed.init_process_group")
    @patch("zmq.Context")
    def test_backend_selection(
        self, mock_zmq_context, mock_init_pg, mock_cuda_available
    ):
        """Test backend selection based on CUDA availability"""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context

        # Test CUDA available
        mock_cuda_available.return_value = True
        with patch("torch.cuda.set_device") as mock_set_device:
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)
            mock_set_device.assert_called_once_with(0)
            # Backend should be nccl (checked in init_process_group call)

        # Test CUDA not available
        mock_cuda_available.return_value = False
        worker = DistributedWorker(0, 2, "localhost", "12345", 12346)
        # Backend should be gloo (checked in init_process_group call)

    def test_execute_code_success(self):
        """Test successful code execution"""
        with patch("torch.distributed.init_process_group"), patch("zmq.Context"):
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Test simple code execution
            result = worker._execute_code("x = 5\nprint(f'Value: {x}')")

            assert result["status"] == "success"
            assert result["rank"] == 0
            assert "Value: 5" in result["output"]

            # Check variable was stored in namespace
            assert worker.namespace["x"] == 5

    def test_execute_code_error(self):
        """Test code execution with errors"""
        with patch("torch.distributed.init_process_group"), patch("zmq.Context"):
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Test code with syntax error
            result = worker._execute_code("x = ")

            assert "error" in result
            assert "traceback" in result
            assert result["rank"] == 0

    def test_execute_code_with_torch(self):
        """Test code execution with torch operations"""
        with patch("torch.distributed.init_process_group"), patch("zmq.Context"):
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Test torch tensor creation
            code = """
import torch
tensor = torch.randn(2, 3)
print(f"Tensor shape: {tensor.shape}")
"""
            result = worker._execute_code(code)

            assert result["status"] == "success"
            assert "Tensor shape: torch.Size([2, 3])" in result["output"]
            assert "tensor" in worker.namespace
            assert isinstance(worker.namespace["tensor"], torch.Tensor)

    def test_get_variable_success(self):
        """Test getting variables from namespace"""
        with patch("torch.distributed.init_process_group"), patch("zmq.Context"):
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Set up test variables
            worker.namespace["test_var"] = 42
            worker.namespace["test_tensor"] = torch.tensor([1, 2, 3])

            # Test regular variable
            result = worker._get_variable("test_var")
            assert result["value"] == 42

            # Test tensor variable
            result = worker._get_variable("test_tensor")
            assert "value" in result
            assert "device" in result
            assert "dtype" in result
            assert "shape" in result
            assert result["shape"] == [3]

    def test_get_variable_not_found(self):
        """Test getting non-existent variable"""
        with patch("torch.distributed.init_process_group"), patch("zmq.Context"):
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            result = worker._get_variable("nonexistent")
            assert "error" in result
            assert "not found" in result["error"]

    def test_set_variable(self):
        """Test setting variables in namespace"""
        with patch("torch.distributed.init_process_group"), patch("zmq.Context"):
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Test setting variable
            result = worker._set_variable({"name": "test_var", "value": 100})
            assert result["status"] == "success"
            assert worker.namespace["test_var"] == 100

    @patch("torch.distributed.destroy_process_group")
    def test_shutdown(self, mock_destroy_pg):
        """Test worker shutdown"""
        with (
            patch("torch.distributed.init_process_group"),
            patch("zmq.Context") as mock_zmq_context,
        ):
            mock_context = Mock()
            mock_socket = Mock()
            mock_context.socket.return_value = mock_socket
            mock_zmq_context.return_value = mock_context

            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)
            worker.shutdown()

            # Check cleanup was called
            mock_destroy_pg.assert_called_once()
            mock_socket.close.assert_called_once()
            mock_context.term.assert_called_once()


class MockZMQSocket:
    """Mock ZMQ socket for testing worker message loop"""

    def __init__(self):
        self.messages = []
        self.responses = []
        self.closed = False

    def recv(self):
        if self.messages:
            return self.messages.pop(0)
        raise zmq.Again()

    def send(self, data):
        self.responses.append(data)

    def setsockopt(self, option, value):
        pass

    def connect(self, address):
        pass

    def close(self):
        self.closed = True

    def add_message(self, message):
        self.messages.append(pickle.dumps(message))


class TestDistributedWorkerMessageLoop:
    """Test worker message processing loop"""

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.barrier")
    def test_message_loop_execute(self, mock_barrier, mock_init_pg):
        """Test message loop with execute message"""
        with patch("zmq.Context") as mock_zmq_context:
            mock_context = Mock()
            mock_socket = MockZMQSocket()
            mock_context.socket.return_value = mock_socket
            mock_zmq_context.return_value = mock_context

            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Add execute message
            execute_msg = Message(
                msg_id="test-123",
                msg_type="execute",
                rank=-1,
                data="x = 42\nprint(x)",
                timestamp=time.time(),
            )
            mock_socket.add_message(execute_msg)

            # Add shutdown message to stop loop
            shutdown_msg = Message(
                msg_id="shutdown-123",
                msg_type="shutdown",
                rank=-1,
                data={},
                timestamp=time.time(),
            )
            mock_socket.add_message(shutdown_msg)

            # Run message loop
            worker.run()

            # Check responses
            assert (
                len(mock_socket.responses) == 2
            )  # execute response + shutdown response

            # Check execute response
            response_data = pickle.loads(mock_socket.responses[0])
            assert response_data.msg_type == "response"
            assert response_data.rank == 0
            assert response_data.data["status"] == "success"
            assert "42" in response_data.data["output"]

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.barrier")
    def test_message_loop_sync(self, mock_barrier, mock_init_pg):
        """Test message loop with sync message"""
        with patch("zmq.Context") as mock_zmq_context:
            mock_context = Mock()
            mock_socket = MockZMQSocket()
            mock_context.socket.return_value = mock_socket
            mock_zmq_context.return_value = mock_context

            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Add sync message
            sync_msg = Message(
                msg_id="sync-123",
                msg_type="sync",
                rank=-1,
                data={},
                timestamp=time.time(),
            )
            mock_socket.add_message(sync_msg)

            # Add shutdown message
            shutdown_msg = Message(
                msg_id="shutdown-123",
                msg_type="shutdown",
                rank=-1,
                data={},
                timestamp=time.time(),
            )
            mock_socket.add_message(shutdown_msg)

            # Run message loop
            worker.run()

            # Check that barrier was called
            mock_barrier.assert_called_once()

            # Check response
            response_data = pickle.loads(mock_socket.responses[0])
            assert response_data.data["status"] == "synced"

    @patch("torch.distributed.init_process_group")
    def test_message_loop_unknown_message(self, mock_init_pg):
        """Test message loop with unknown message type"""
        with patch("zmq.Context") as mock_zmq_context:
            mock_context = Mock()
            mock_socket = MockZMQSocket()
            mock_context.socket.return_value = mock_socket
            mock_zmq_context.return_value = mock_context

            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Add unknown message
            unknown_msg = Message(
                msg_id="unknown-123",
                msg_type="unknown_type",
                rank=-1,
                data={},
                timestamp=time.time(),
            )
            mock_socket.add_message(unknown_msg)

            # Add shutdown message
            shutdown_msg = Message(
                msg_id="shutdown-123",
                msg_type="shutdown",
                rank=-1,
                data={},
                timestamp=time.time(),
            )
            mock_socket.add_message(shutdown_msg)

            # Run message loop
            worker.run()

            # Check error response
            response_data = pickle.loads(mock_socket.responses[0])
            assert "error" in response_data.data
            assert "Unknown message type" in response_data.data["error"]


class TestDistributedWorkerIntegration:
    """Integration tests for worker functionality"""

    @pytest.mark.slow
    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.destroy_process_group")
    def test_worker_lifecycle(self, mock_destroy_pg, mock_init_pg):
        """Test complete worker lifecycle"""
        with patch("zmq.Context") as mock_zmq_context:
            mock_context = Mock()
            mock_socket = Mock()
            mock_context.socket.return_value = mock_socket
            mock_zmq_context.return_value = mock_context

            # Create worker
            worker = DistributedWorker(0, 2, "localhost", "12345", 12346)

            # Test code execution
            result = worker._execute_code("import torch\nt = torch.randn(3, 3)")
            assert result["status"] == "success"

            # Test variable access
            result = worker._get_variable("t")
            assert "value" in result
            assert result["shape"] == [3, 3]

            # Test shutdown
            worker.shutdown()
            mock_destroy_pg.assert_called_once()
