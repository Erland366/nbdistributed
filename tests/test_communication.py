"""
Tests for communication.py
"""

import pytest
import time
import threading
import zmq
import pickle

from jupyter_distributed.communication import Message, CommunicationManager


class TestMessage:
    """Test the Message dataclass"""

    def test_message_creation(self):
        """Test creating a message"""
        msg = Message(
            msg_id="test-123",
            msg_type="execute",
            rank=0,
            data="test_data",
            timestamp=time.time(),
        )

        assert msg.msg_id == "test-123"
        assert msg.msg_type == "execute"
        assert msg.rank == 0
        assert msg.data == "test_data"
        assert isinstance(msg.timestamp, float)

    def test_message_serialization(self):
        """Test message can be pickled/unpickled"""
        msg = Message(
            msg_id="test-123",
            msg_type="execute",
            rank=0,
            data={"code": "print('hello')"},
            timestamp=time.time(),
        )

        serialized = pickle.dumps(msg)
        deserialized = pickle.loads(serialized)

        assert deserialized.msg_id == msg.msg_id
        assert deserialized.msg_type == msg.msg_type
        assert deserialized.rank == msg.rank
        assert deserialized.data == msg.data


class TestCommunicationManager:
    """Test the CommunicationManager class"""

    def test_initialization(self, free_port):
        """Test communication manager initialization"""
        manager = CommunicationManager(num_ranks=2, base_port=free_port)

        assert manager.num_ranks == 2
        assert manager.base_port == free_port
        assert manager.context is not None
        assert manager.coordinator_socket is not None
        assert manager.running is True
        assert manager.handler_thread.is_alive()

        manager.shutdown()

    def test_message_handler_thread(self, free_port):
        """Test that message handler thread starts and stops properly"""
        manager = CommunicationManager(num_ranks=2, base_port=free_port)

        # Thread should be running
        assert manager.handler_thread.is_alive()

        # Shutdown should stop the thread
        manager.shutdown()
        manager.handler_thread.join(timeout=1)
        assert not manager.handler_thread.is_alive()

    def test_timeout_handling(self, free_port):
        """Test timeout when no workers respond"""
        manager = CommunicationManager(num_ranks=2, base_port=free_port)

        # This should timeout since no workers are connected
        with pytest.raises(TimeoutError):
            manager.send_to_all("test", {}, timeout=0.1)

        manager.shutdown()


class MockWorker:
    """Mock worker for testing communication"""

    def __init__(self, rank, port):
        self.rank = rank
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, f"worker_{rank}".encode())
        self.socket.connect(f"tcp://localhost:{port}")
        self.running = False

    def start(self):
        """Start mock worker response loop"""
        self.running = True
        self.thread = threading.Thread(target=self._response_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop mock worker"""
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)
        self.socket.close()
        self.context.term()

    def _response_loop(self):
        """Mock response loop"""
        while self.running:
            try:
                if self.socket.poll(100):  # 100ms timeout
                    message_data = self.socket.recv()
                    message = pickle.loads(message_data)

                    # Create response
                    response = Message(
                        msg_id=message.msg_id,
                        msg_type="response",
                        rank=self.rank,
                        data={
                            "output": f"Response from rank {self.rank}",
                            "status": "success",
                        },
                        timestamp=time.time(),
                    )

                    self.socket.send(pickle.dumps(response))
            except zmq.Again:
                continue
            except Exception as e:
                print(f"Mock worker {self.rank} error: {e}")
                break


class TestCommunicationIntegration:
    """Integration tests with mock workers"""

    def test_send_to_all_with_workers(self, free_port):
        """Test sending messages to all workers"""
        manager = CommunicationManager(num_ranks=2, base_port=free_port)

        # Start mock workers
        workers = [MockWorker(0, free_port), MockWorker(1, free_port)]
        for worker in workers:
            worker.start()

        time.sleep(0.1)  # Let workers connect

        try:
            # Send message to all workers
            responses = manager.send_to_all("execute", "print('hello')", timeout=2.0)

            assert len(responses) == 2
            assert 0 in responses
            assert 1 in responses
            assert responses[0]["status"] == "success"
            assert responses[1]["status"] == "success"

        finally:
            # Cleanup
            for worker in workers:
                worker.stop()
            manager.shutdown()

    def test_send_to_specific_ranks(self, free_port):
        """Test sending messages to specific ranks"""
        manager = CommunicationManager(num_ranks=3, base_port=free_port)

        # Start mock workers
        workers = [MockWorker(i, free_port) for i in range(3)]
        for worker in workers:
            worker.start()

        time.sleep(0.1)  # Let workers connect

        try:
            # Send to ranks 0 and 2 only
            responses = manager.send_to_ranks(
                [0, 2], "execute", "test_code", timeout=2.0
            )

            assert len(responses) == 2
            assert 0 in responses
            assert 2 in responses
            assert 1 not in responses

        finally:
            # Cleanup
            for worker in workers:
                worker.stop()
            manager.shutdown()

    def test_send_to_single_rank(self, free_port):
        """Test sending message to a single rank"""
        manager = CommunicationManager(num_ranks=2, base_port=free_port)

        # Start mock workers
        workers = [MockWorker(0, free_port), MockWorker(1, free_port)]
        for worker in workers:
            worker.start()

        time.sleep(0.1)  # Let workers connect

        try:
            # Send to rank 1 only
            response = manager.send_to_rank(1, "execute", "test_code", timeout=2.0)

            assert response["status"] == "success"
            assert "Response from rank 1" in response["output"]

        finally:
            # Cleanup
            for worker in workers:
                worker.stop()
            manager.shutdown()


class TestCommunicationErrorHandling:
    """Test error handling in communication"""

    def test_invalid_rank_specification(self, free_port):
        """Test handling of invalid rank specifications"""
        manager = CommunicationManager(num_ranks=2, base_port=free_port)

        try:
            # Should handle empty rank list gracefully
            with pytest.raises(TimeoutError):  # No workers to respond
                manager.send_to_ranks([], "execute", "test", timeout=0.1)

            # Should handle out-of-range ranks
            with pytest.raises(TimeoutError):  # No workers to respond
                manager.send_to_ranks([5], "execute", "test", timeout=0.1)

        finally:
            manager.shutdown()

    def test_partial_worker_failure(self, free_port):
        """Test handling when some workers fail to respond"""
        manager = CommunicationManager(num_ranks=2, base_port=free_port)

        # Start only one worker
        worker = MockWorker(0, free_port)
        worker.start()

        time.sleep(0.1)

        try:
            # This should timeout since rank 1 never responds
            with pytest.raises(TimeoutError):
                manager.send_to_all("execute", "test", timeout=0.5)

        finally:
            worker.stop()
            manager.shutdown()
