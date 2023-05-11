import argparse
from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx,  numpy as np
from numbers import Number
import math
#import matplotlib.pyplot as plt
import pandas as pd
import random, time
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit, NormalizeFeatures
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix
import sklearn.datasets as datasets
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

sys.path.append('../')
from data_utils import *
from edge_prediction import edge_prediction
from graph_utils import get_weights, transform_edge_weights
from label_classification import label_classification
from models.baseline_models import *
from models.train_models import *
from evaluation_metric import *
from train_utils import *
from graph_utils import *
from simulation_utils import *
from umap_functions import *
from SBM.read_SBM import *


def data_set(name, n_samples = 500, n_neighbours = 50,features = 'none', standardize = True, 
         centers = 4, cluster_std = [0.1,0.1,1.0,1.0],
         factor = 0.2, noise = 0.05,
         random_state = 0, radius = False, epsilon = 0.5, 
         SBMtype = 'lazy'):

    if name == 'Blob':
        X, y_true = datasets.make_blobs(
                    n_samples= n_samples, centers=centers, cluster_std=cluster_std,
                    random_state=random_state)
        G = convert_to_graph(X, n_neighbours = n_neighbours,features=features,standardize=standardize, eps = radius, epsilon = epsilon)
        G.y = torch.from_numpy(y_true)
    elif name == 'Sphere':
        X, y_true = create_sphere(r = 1)
        G = convert_to_graph(X, n_neighbours = n_neighbours,features=features,standardize=standardize, eps = radius, epsilon = epsilon)
        G.y = torch.from_numpy(y_true)
    elif name == 'Circles':
        X, y_true = datasets.make_circles(
            n_samples=n_samples, factor=factor, noise=noise, random_state=0
        )
        G = convert_to_graph(X, n_neighbours = n_neighbours,features=features,standardize=standardize, eps = radius, epsilon = epsilon)
        G.y = torch.from_numpy(y_true)
    elif name == 'Moons':
        X, y_true = datasets.make_moons(
            n_samples=n_samples, noise=noise, random_state=random_state
        )
        G = convert_to_graph(X, n_neighbours = n_neighbours,features=features,standardize=standardize, eps = radius, epsilon = epsilon)
        G.y = torch.from_numpy(y_true)
    elif name == 'Swissroll':
        X, y_true = datasets.make_swiss_roll(n_samples = n_samples, noise = noise, random_state=random_state)
        G = convert_to_graph(X, n_neighbours = n_neighbours,features=features,standardize=standardize, eps = radius, epsilon = epsilon)
        G.y = torch.from_numpy(y_true)

    elif name == 'Scurve':
        X, y_true = datasets.make_s_curve(n_samples = n_samples, random_state=random_state)      
        G = convert_to_graph(X, n_neighbours = n_neighbours,features=features,standardize=standardize, eps = radius, epsilon = epsilon)
        G.y = torch.from_numpy(y_true)
    elif name == 'SBM':
        X,y_true, G = readSBM(type = SBMtype, features = None)  

    elif name == 'Cora':
        dataset = Planetoid(root='Planetoid', name='Cora', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        X, y_true = G.x, G.y

    elif name == 'Pubmed':
        dataset = Planetoid(root='Planetoid', name='Pubmed', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        X, y_true = G.x, G.y

    elif name == 'Citeseer':
        dataset = Planetoid(root='Planetoid', name='Citeseer', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        X, y_true = G.x, G.y
    else:
        raise ValueError("Model unknown!!")

    return(X, y_true, G)
