import os
import sys

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset, DataLoader, TensorDataset

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from module import * 
from utils import *


if __name__ == '__main__': 
    model = ViViT(in_channels = 1, num_frames=75, L=1, drop=0.2)
    
    model.load_state_dict({k.replace('module.', '') : v for k, v in torch.load('model_0.pt').items()})
    
    model.eval() 
    
    x = torch.randn(1, 75, 1, 224, 224) 
    y = torch.randn(1, 64) 
    
    out = model(x, y)
    print(out)
    