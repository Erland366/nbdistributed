"""
Jupyter Distributed Extension for PyTorch
Enables interactive distributed training in Jupyter notebooks
"""


def load_ipython_extension(ipython):
    """Load the extension in IPython/Jupyter"""
    from .magic import DistributedMagic

    # Clean up any existing state
    DistributedMagic._process_manager = None
    DistributedMagic._comm_manager = None
    DistributedMagic._num_processes = 0

    # Create an instance and register all magics
    magic = DistributedMagic(ipython)
    ipython.register_magics(magic)


def unload_ipython_extension(ipython):
    """Unload the extension"""
    from .magic import DistributedMagic

    DistributedMagic.shutdown_all()
