import os
import sys
import copy

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split 

import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def demo_basic(rank, world_size):
   
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    print(f"rank: {rank}", torch.cuda.mem_get_info(rank))

    cleanup()  
    
if __name__ == '__main__':  

    world_size = torch.cuda.device_count()
    mp.spawn(demo_basic,
             args=[world_size],
             nprocs=world_size,
             join=True)