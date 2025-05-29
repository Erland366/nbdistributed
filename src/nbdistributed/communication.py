"""
Communication layer for distributed processes
"""

import zmq
import pickle
import threading
import time
from typing import Any, Dict, List
from dataclasses import dataclass
import uuid


@dataclass
class Message:
    msg_id: str
    msg_type: str
    rank: int
    data: Any
    timestamp: float


class CommunicationManager:
    def __init__(self, num_processes: int, base_port: int = 5555):
        self.num_processes = num_processes
        self.base_port = base_port
        self.context = zmq.Context()

        # Main process acts as coordinator
        self.coordinator_socket = self.context.socket(zmq.ROUTER)
        self.coordinator_socket.bind(f"tcp://*:{base_port}")

        self.worker_sockets = {}
        self.message_queue = {}
        self.response_events = {}

        # Start message handler thread
        self.running = True
        self.handler_thread = threading.Thread(target=self._message_handler)
        self.handler_thread.daemon = True
        self.handler_thread.start()

    def _message_handler(self):
        """Handle incoming messages from workers"""
        while self.running:
            try:
                if self.coordinator_socket.poll(100):  # 100ms timeout
                    identity, message = self.coordinator_socket.recv_multipart()
                    msg = pickle.loads(message)

                    if msg.msg_id not in self.message_queue:
                        self.message_queue[msg.msg_id] = {}

                    self.message_queue[msg.msg_id][msg.rank] = msg

                    # Signal if we have all responses
                    if (
                        msg.msg_id in self.response_events
                        and len(self.message_queue[msg.msg_id]) == self.num_processes
                    ):
                        self.response_events[msg.msg_id].set()

            except zmq.Again:
                continue
            except Exception as e:
                print(f"Error in message handler: {e}")

    def send_to_all(
        self, msg_type: str, data: Any, timeout: float = 30.0
    ) -> Dict[int, Any]:
        """Send message to all workers and wait for responses"""
        msg_id = str(uuid.uuid4())
        message = Message(
            msg_id=msg_id,
            msg_type=msg_type,
            rank=-1,  # From coordinator
            data=data,
            timestamp=time.time(),
        )

        # Set up response collection
        self.response_events[msg_id] = threading.Event()

        # Send to all workers
        serialized = pickle.dumps(message)
        for rank in range(self.num_processes):
            worker_id = f"worker_{rank}".encode()
            self.coordinator_socket.send_multipart([worker_id, serialized])

        # Wait for all responses
        if self.response_events[msg_id].wait(timeout):
            responses = self.message_queue[msg_id]
            del self.message_queue[msg_id]
            del self.response_events[msg_id]
            return {rank: msg.data for rank, msg in responses.items()}
        else:
            raise TimeoutError(f"Timeout waiting for responses to {msg_id}")

    def send_to_rank(
        self, rank: int, msg_type: str, data: Any, timeout: float = 30.0
    ) -> Any:
        """Send message to specific rank"""
        responses = self.send_to_ranks([rank], msg_type, data, timeout)
        return responses[rank]

    def send_to_ranks(
        self, ranks: List[int], msg_type: str, data: Any, timeout: float = 30.0
    ) -> Dict[int, Any]:
        """Send message to specific ranks"""
        msg_id = str(uuid.uuid4())
        message = Message(
            msg_id=msg_id, msg_type=msg_type, rank=-1, data=data, timestamp=time.time()
        )

        self.response_events[msg_id] = threading.Event()

        serialized = pickle.dumps(message)
        for rank in ranks:
            worker_id = f"worker_{rank}".encode()
            self.coordinator_socket.send_multipart([worker_id, serialized])

        # Modified wait condition for subset of ranks
        start_time = time.time()
        while time.time() - start_time < timeout:
            if msg_id in self.message_queue and len(self.message_queue[msg_id]) == len(
                ranks
            ):
                responses = self.message_queue[msg_id]
                del self.message_queue[msg_id]
                del self.response_events[msg_id]
                return {rank: msg.data for rank, msg in responses.items()}
            time.sleep(0.01)

        raise TimeoutError(f"Timeout waiting for responses from ranks {ranks}")

    def shutdown(self):
        """Shutdown communication"""
        self.running = False
        self.handler_thread.join()
        self.coordinator_socket.close()
        self.context.term()
