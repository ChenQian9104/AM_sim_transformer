import os 
import numpy as np 


def load_pointwise_simulation_data(path='./SimulationData/compressor/'):

    npy_file_list = os.listdir(path)
    data = None

    for npy_file in npy_file_list: 
        file_name = path + npy_file_list
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

    return v, data[:, -1]

