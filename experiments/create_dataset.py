import argparse
import copy, collections
import numpy as np
from numbers import Number
import math
import sys, os

import torch_geometric.transforms as T
from torch_geometric.data import Data
import sklearn.datasets as datasets
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

sys.path.append('../')
from data_utils import *
from graph_utils import convert_to_graph
from experiments.SBM.read_SBM import *
from simulation_utils import *


def create_dataset(name, n_samples = 500, n_neighbours = 50, features='none', 
                   standardize=True, centers = 4, cluster_std = [0.1,0.1,1.0,1.0],
                   factor = 0.2, noise = 0.05, random_state = 0, radius_knn = 0, bw = 1,
                   SBMtype = 'lazy', nb_loops=5, radius_tube=4, radius_torus=10):

    if name == 'Blob':
        X, y_true = datasets.make_blobs( n_samples=n_samples, centers=centers, 
                                        cluster_std=cluster_std, random_state=random_state)
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(y_true)
        X_true = np.nan

    elif name == 'Sphere':
        X, y_true = create_sphere(r = 1, size = n_samples)
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(y_true)
        X_true = np.nan

    elif name == 'Circles':
        X, y_true = datasets.make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=0)
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(y_true)
        X_true = np.nan

    elif name == 'Moons':
        X, y_true = datasets.make_moons( n_samples=n_samples, noise=noise, random_state=random_state)
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(y_true)
        X_true = np.nan

    elif name == 'Swissroll':
        X, y_true = datasets.make_swiss_roll(n_samples = n_samples, noise = noise, random_state=random_state)
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(y_true)
        X_true = np.nan

    elif name == 'Scurve':
        X, y_true = datasets.make_s_curve(n_samples = n_samples, random_state=random_state)      
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(y_true)
        X_true = np.nan

    elif name == "Trefoil":
        t = np.linspace(0, 2 * np.pi, n_samples)
        x = np.sin(t) + 2 * np.sin(2 * t)
        y = np.cos(t) - 2 * np.cos(2 * t)
        z = -np.sin(3 * t)
        X = np.vstack([x, y, z]).T
        X_true = np.vstack([np.sin(t) , np.cos(t) ]).T
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(t)
        y_true = G.y
        
    elif name == "Helix":
        t = np.linspace(0, 2. * np.pi, n)
        x = (radius_torus + radius_tube * np.cos(nb_loops * t)) * np.cos(t)
        y = (radius_torus + radius_tube * np.cos(nb_loops * t)) * np.sin(t)
        z = radius_tube * np.sin(nb_loops * t)
        X = np.vstack([x, y, z]).T
        X_true = np.vstack([x, y]).T
        G = convert_to_graph(X, n_neighbours=n_neighbours, features=features, standardize=standardize, 
                             radius_knn = radius_knn, bw = bw)
        G.y = torch.from_numpy(t)
        y_true = G.y
             
    elif name == 'SBM':
        ### needs to be automated
        X, y_true, G = readSBM(type = SBMtype, features = features)
        X_true = np.nan 

    elif name == 'Cora':
        dataset = Planetoid(root='Planetoid', name='Cora', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        X, y_true = G.x, G.y
        X_true = X

    elif name == 'Pubmed':
        dataset = Planetoid(root='Planetoid', name='Pubmed', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        X, y_true = G.x, G.y
        X_true = X

    elif name == 'Citeseer':
        dataset = Planetoid(root='Planetoid', name='Citeseer', transform=NormalizeFeatures())
        G = dataset[0]  # Get the first graph object.
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        X, y_true = G.x, G.y
        X_true = X

    else:
        raise ValueError("Data unknown!!")
    
    return(X, X_true, y_true, G)
