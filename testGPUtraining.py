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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def demo_basic(rank, world_size, dataset):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    # Prepare dataloader on each GPU
    train_data_loader = DataLoader(dataset, 
                              batch_size = 32, 
                              pin_memory=True, 
                              shuffle=False, 
                              sampler=sampler )
    """                          
    for data, label in data_loader: 
        print(f"on rank: {rank}, | shape of data: ", data.shape)
        print(f"on rank: {rank}    ", label)
    """                         
                             

    # create model and move it to GPU with id rank
    
    model = MultiHeadAttention().to(rank)
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(ddp_model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size = 10, gamma=0.95)

    num_epoch = 100

    for epoch in range(num_epoch + 1): 

        train_loss = 0 
        
        train_data_loader.sampler.set_epoch(epoch)
        
        dist.barrier()  
        
        for data, label in train_data_loader: 
            
                      
            batch_size = data.shape[0]

            input = data.to(rank)
            label = label.to(rank) 


            pred = ddp_model(input) 
            loss = criterion(pred, label) 

            optimizer.zero_grad() 
            loss.backward() 

            optimizer.step() 

            train_loss += loss.item()/len(train_data_loader)
            
        scheduler.step()
        if rank == 0: 
            print(f"rank: {rank} | epoch: {epoch} | loss: {loss.item()}")

       

    cleanup()    



if __name__ == '__main__':


    # Read these setup from config file 
    batch_size = 32

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    print('Available memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')

    image = torch.randn(4, 16, 196, 768) 
    label = torch.randn(4, 16, 196, 768) 
   
    dataset = TensorDataset(image, label)



    # Train the model 
    
    world_size = torch.cuda.device_count()
    mp.spawn(demo_basic,
             args=(world_size,dataset),
             nprocs=world_size,
             join=True)



