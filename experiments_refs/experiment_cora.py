import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeature


from data_utils import *
dataset_name = 'Cora'
dataset = Planetoid(root='Planetoid', name=dataset_name, transform=NormalizeFeatures())
data = dataset[0]
dataset_print(dataset)
data_print(data)
