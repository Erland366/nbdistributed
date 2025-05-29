"""
Pytest configuration and shared fixtures
"""

import pytest
import tempfile
import sys
import time
import socket
from pathlib import Path

# Add src to Python path for testing
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from jupyter_distributed.communication import CommunicationManager, Message
from jupyter_distributed.process_manager import ProcessManager


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def free_port():
    """Get a free port for testing"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def sample_message():
    """Create a sample message for testing"""
    return Message(
        msg_id="test-123",
        msg_type="execute",
        rank=0,
        data="print('hello world')",
        timestamp=time.time(),
    )


@pytest.fixture
def comm_manager(free_port):
    """Create a communication manager for testing"""
    manager = CommunicationManager(num_ranks=2, base_port=free_port)
    yield manager
    manager.shutdown()


@pytest.fixture
def process_manager():
    """Create a process manager for testing"""
    manager = ProcessManager()
    yield manager
    manager.shutdown()


@pytest.fixture(scope="session")
def mock_jupyter_kernel():
    """Mock Jupyter kernel environment for testing magic commands"""
    try:
        from IPython.testing import tools as tt
        from IPython.core.interactiveshell import InteractiveShell

        # Create a test shell
        shell = InteractiveShell.instance()
        shell.magic("load_ext jupyter_distributed")

        yield shell

        # Cleanup
        try:
            shell.magic("dist_shutdown")
        except:
            pass

    except ImportError:
        pytest.skip("IPython not available for kernel testing")


@pytest.fixture
def mock_torch_env(monkeypatch):
    """Mock torch distributed environment variables"""
    env_vars = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "2",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars
