"""
Helpers for single-GPU training (Colab/T4 compatible).
"""
import torch as th
import torch.distributed as dist

def setup_dist():
    """Setup for single GPU - no distributed training needed."""
    # 在单GPU环境中，不需要初始化分布式进程组
    pass

def dev():
    """Get the device to use - always cuda:0 if available."""
    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """Load a PyTorch file."""
    return th.load(path, **kwargs)

def sync_params(params):
    """No parameter synchronization needed in single GPU."""
    pass
