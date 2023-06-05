import os 
import numpy as np 
import scipy.io
import torch
from torch.utils.data import random_split, Dataset, DataLoader


def load_pointwise_simulation_data(path='./SimulationData/compressor/'):

    npy_file_list = os.listdir(path)
    data = None

    for npy_file in npy_file_list: 
        file_name = path + npy_file
        temp = np.load(file_name)
        if data is None: 
            data = temp 
        else: 
            data = np.concatenate((data, temp), axis=0)
    
    return data

def normalize_simulation_data(data, img_size=224, img_domain=200, scale_height=40, scale_time=10000, scale_temperature=100):

    """
    Args: 
        data: a n x 5 matrix; each row represent a single point (x, y, z), and its temperature, temp, at time t
              (x, y, z, t, temp)

    """ 

    # Transfer the x & t coordinate to pixel index of the image, then divide it by image size
    dx = img_domain / (img_size - 1)

    data[:, 0] = (data[:, 0] + img_domain/2) // dx / img_size 
    data[:, 1] = (data[:, 1] + img_domain/2) // dx / img_size 

    # Normalize the height (along z direction)
    data[:, 2] = data[:, 2]/scale_height

    # Normalize the time 
    data[:, 3] = data[:, 3]/scale_time 

    # Normalize the temperature 
    data[:, 4] = data[:, 4]/scale_temperature

    return data 
    
def query_position_encoding(data, freq_num=8, query_dim=4): 

    """
    Map the (x, y, z, t) query vector to high dimensional space (freq_num*query_dim*2)

    x -> sin(2^0*pi*x), cos(2^0*pi*x), sin(2^1*pi*x), cos(2^1*pi*x), ......, sin(2^(L-1)*pi*x), cos(2^(L-1)*pi*x)
    """

    v = np.zeros((data.shape[0], 2*freq_num*query_dim))

    query_v = data[:, :-1]*np.pi

    for L in range(freq_num): 

        v_sin = np.sin(2**L*query_v)
        v_cos = np.cos(2**L*query_v)

        v[:, 2*L : : 2*freq_num] = v_sin 
        v[:, 2*L + 1 : : 2*freq_num] = v_cos

    return np.concatenate((v, data[:, -1].reshape(data.shape[0], 1)), axis=1)
    
def load_data(path = './SimulationData/'): 

    geometry_file_list = os.listdir(path) 
    
    data_list = []
    
    
    for geometry_file in geometry_file_list: 
        file_path = path + geometry_file + '/'
        data = load_pointwise_simulation_data(file_path) 
        data = normalize_simulation_data(data) 
        data = query_position_encoding(data) 
        data_list.append(data)
        
    return data_list
        
        
    

def generateDataset(data, ratio = [0.7, 0.2, 0.1], batch_size=4): 
    n = data.shape[0] 

    n_train = int(n * ratio[0])
    n_test =  int(n * ratio[1])
    n_val  = n - n_train - n_test

    train_data, test_data, val_data = random_split(data, [n_train, n_test, n_val])
    
    return train_data, test_data, val_data


def generateDataLoader(data, ratio = [0.7, 0.2, 0.1], batch_size=4): 

    n = data.shape[0] 

    n_train = int(n * ratio[0])
    n_test =  int(n * ratio[1])
    n_val  = n - n_train - n_test

    train_data, test_data, val_data = random_split(data, [n_train, n_test, n_val])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    validation_dataloader=DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, validation_dataloader

def loadGeoImage(file_name='./Geometry/compressor.mat'): 


    mat = scipy.io.loadmat(file_name) 
    val = np.array(mat['result'], dtype = float)
    image = torch.from_numpy(val).type(torch.float32)

    image = torch.permute(image,(2, 0, 1))
    image = torch.unsqueeze(image, 1)
    return image
    
    
def load_data(path_simualtion_data = './SimulationData/', path_geometry_data = './Geometry/'): 

    geometry_file_list = os.listdir(path_simualtion_data) 
    
    data_list = []
    image_list = []
    
    key_to_image = {}
    
    key = 0 
    for geometry_file in geometry_file_list: 
        file_path_npy = path_simualtion_data + geometry_file + '/'
        data = load_pointwise_simulation_data(file_path_npy) 
        data = normalize_simulation_data(data) 
        data = query_position_encoding(data) 
        
        
        keys = np.ones((data.shape[0], 1))
        data_list.append(np.concatenate((keys, data), axis = 1))
        
        
        
        file_path_mat = path_geometry_data + geometry_file + '.mat' 
        
        image = loadGeoImage(file_path_mat) 
        image_list.append(image)
        
        key_to_image[key] = image
        
        key += 1
        
        
        
        
    return data_list, image_list, key_to_image
    
    
def generateDataLoaderList( ratio = [0.7, 0.2, 0.1], batch_size=4):

    data_list, image_list, key_to_image = load_data()
    
    train_dataloader_list, test_dataloader_list, validation_dataloader_list = [], [], []
    
    for data in data_list: 
        
    
        train_dataloader, test_dataloader, validation_dataloader = generateDataLoader(data, ratio = ratio, batch_size=batch_size)
        
        train_dataloader_list.append(train_dataloader) 
        test_dataloader_list.append(test_dataloader) 
        validation_dataloader_list.append(validation_dataloader)
        
    return train_dataloader_list, test_dataloader_list, validation_dataloader_list, image_list, key_to_image
        
        
   