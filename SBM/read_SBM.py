import pandas as pd
from torch_geometric.data import Data
import torch
import sys, os
# sys.path.append('../')

def readSBM(type = 'lazy', features = None):
    path = os.path.dirname(os.getcwd()) +'/SBM/'
    if type == 'lazy':
        data = pd.read_csv(path + "lazySBM.csv")
        y = pd.read_csv(path + "lazy_y.csv")
    elif type == 'dense':
        data = pd.read_csv(path + "denseSBM.csv")
        y = pd.read_csv(path + "dense_y.csv")
    elif type == 'custom':
        data = pd.read_csv(path + "customSBM.csv")
        y = pd.read_csv(path + "custom_y.csv")
    else:
        print('Type Unknown!')
    
    if features == 'ones':
        new_data = Data(x=torch.ones(500, 500), y = torch.Tensor(y.values).T, edge_index=torch.Tensor(data.values).T)
    else:
        new_data = Data(x=torch.eye(500), y = torch.Tensor(y.values).T, edge_index=torch.Tensor(data.values).T)

    return new_data.x, new_data.y, new_data
