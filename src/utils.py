"""
Utility functions: seeding for reproducibility, helpers for training loops.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set seeds for all sources of randomness to ensure reproducibility.
    
    Covers:
    - Python's `random`
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuDNN deterministic mode
    - PYTHONHASHSEED env variable
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_averager():
    """
    Returns a closure that maintains a running mean.
    
    Usage:
        avg = make_averager()
        for x in values:
            current_mean = avg(x)
        final_mean = avg()  # returns mean without updating
    """
    count = 0
    total = 0.0

    def averager(new_value=None):
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager


def get_device():
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")