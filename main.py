import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset, DataLoader

from module import * 
from utils import *

if __name__ == '__main__':


    # Read these setup from config file 
    batch_size = 32

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


    train_dataloader, test_dataloader, validation_dataloader = generateDataLoader(data[:100000, :].astype(np.float32), ratio = [0.8, 0.1, 0.1], batch_size=batch_size)

    # Train the model 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ViViT(in_channels = 1, num_frames=num_layers, L=1, drop=0.2)
    model.to(device) 

    #image = image.to(device)

    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size = 10, gamma=0.95)

    num_epoch = 100

    for epoch in range(num_epoch + 1): 

        train_loss = 0 

        for data in train_dataloader: 
            
            batch_size = data[:, :-1].shape[0]

            input = data[:, :-1].to(device)
            label = data[:, -1:].to(device) 
            
            image_ = image.repeat((batch_size, 1, 1, 1, 1)).to(device)

            pred = model(image_, input) 
            loss = criterion(pred, label) 

            optimizer.zero_grad() 
            loss.backward() 

            optimizer.step() 

            train_loss += loss.item()/len(train_dataloader)
            
        scheduler.step()

        if epoch % 5 == 0: 

            with torch.no_grad(): 

                val_loss = 0 

                for data in validation_dataloader:
                
                    batch_size = data[:, :-1].shape[0]
                    
                    input = data[:, :-1].to(device)
                    label = data[:, -1:].to(device) 
                    
                    image_ = image.repeat((batch_size, 1, 1, 1, 1)).to(device)
                    pred_val = model(image_, input) 
                    loss = criterion(pred_val, label)  
                    val_loss += loss.item()/len(validation_dataloader)
            
            print('epoch: %d, loss: %.5f, validation loss: %.5f' % (epoch, train_loss, val_loss))




    

