import torch 
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np

class Dataset(): 
    def __init__(self):
        self.data =[np.arange(1,4), np.arange(2,8), np.arange(3,6), np.arange(4,7)]
        self.label =np.array([[1],[2],[3],[4]])
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, key): 
        return self.data[key], self.label[key]
        
        
        
def collate_fn_padd(data): 
    max_length = 0
    for d in data: 
        max_length = max(max_length, len(d[0])) 
    print(max_length)
    

    a = [np.array(d[0]) for d in data]
    b = [np.array(d[1]) for d in data] 
    
    x = [d[0] for d in data]
    
    
    padded = [torch.cat([torch.tensor(item), torch.tensor([0]).expand(max_length - len(item))]) for item in x]
    
    batches = torch.cat([item[None] for item in padded])
    return batches, torch.tensor(b) 
    
    
    
if __name__ == '__main__': 
    
    d = Dataset()
    print(len(d))
    print(d[0])
    dataloader = DataLoader(d, batch_size = 2, collate_fn = collate_fn_padd)
    
    for data, label in dataloader: 
    print(data)
    print(label) 
    print()
    