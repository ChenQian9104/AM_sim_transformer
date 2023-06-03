import os
import sys
import copy

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
    
def demo_basic(rank, world_size, dataset, image, validation_dataloader, batch_size=32):
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
    num_layers = image.shape[0]
    model = ViViT(in_channels = 1, num_frames=num_layers, L=1, drop=0.2).to(rank)
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-1, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size = 10, gamma=0.9)

    num_epoch = 5

    for epoch in range(num_epoch + 1): 

        train_loss = 0 
        
        train_data_loader.sampler.set_epoch(epoch)
        
        #dist.barrier()  
        
        for data in train_data_loader: 
            
                      
            batch_size = data[:, :-1].shape[0]

            input = data[:, :-1].to(rank)
            label = data[:, -1:].to(rank) 
            
            image_ = image.repeat((batch_size, 1, 1, 1, 1)).to(rank)

            pred = ddp_model(image_, input) 
            loss = criterion(pred, label) 

            optimizer.zero_grad() 
            loss.backward() 

            optimizer.step() 

            train_loss += loss.item()/len(train_data_loader)
            
        scheduler.step()
        

        if epoch % 1 == 0 and rank == 0: 

            with torch.no_grad(): 

                val_loss = 0 

                for data in validation_dataloader:
                
                    batch_size = data[:, :-1].shape[0]
                    
                    input = data[:, :-1].to(rank)
                    label = data[:, -1:].to(rank) 
                    
                    image_ = image.repeat((batch_size, 1, 1, 1, 1)).to(rank)
                    pred_val = ddp_model(image_, input) 
                    loss = criterion(pred_val, label)  
                    val_loss += loss.item()/len(validation_dataloader)
            
            print('epoch: %d, loss: %.5f, validation loss: %.5f' % (epoch, train_loss, val_loss))
            best_model_wts = copy.deepcopy( ddp_model.state_dict() )
            torch.save( best_model_wts,f"model_{epoch}.pt")


    cleanup()    



if __name__ == '__main__':


    # Read these setup from config file 
    batch_size = 64

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    print('Available memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')

    image = loadGeoImage()
    print('Image tensor shape: ', image.shape)
    num_layers = image.shape[0]

    #image = image.repeat((batch_size, 1, 1, 1, 1))
    #print('Batch Image tensor shape: ', image.shape)


    # Load the simulation data 
    data = load_pointwise_simulation_data()
    
    data = normalize_simulation_data(data)

    data = query_position_encoding(data)
    
    print('The size of simulation data is: ', data.shape)

    np.random.shuffle(data)

    print(data[:100000, -1].sum())
    
    train_dataset, test_dataset, validation_dataset = generateDataset(data[:100000, :].astype(np.float32), ratio = [0.8, 0.1, 0.1], batch_size=batch_size)
    
    validation_dataloader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    dataset = train_dataset 


    # Train the model 
    
    world_size = torch.cuda.device_count()
    mp.spawn(demo_basic,
             args=(world_size,dataset, image, validation_dataloader, batch_size),
             nprocs=world_size,
             join=True)




    

