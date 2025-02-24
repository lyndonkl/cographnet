import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                          rank=rank,
                          world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def run_distributed(fn: Callable, world_size: int, *args, **kwargs):
    """Run a function in distributed mode."""
    mp.spawn(fn,
            args=(world_size, *args),
            nprocs=world_size,
            join=True) 