import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit, NormalizeFeatures
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, from_scipy_sparse_matrix, to_undirected
from scipy.sparse import csr_matrix
import numpy as np
import scipy as sc
import sklearn as sk
import umap
import pickle
import argparse
#from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx,  numpy as np
from numbers import Number
import math
import pandas as pd
import random, time
import sys, os
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from sklearn.datasets import *

sys.path.append('../')
from models.baseline_models import *
from models.train_models import *
from umap_functions import *
from graph_utils import *
from experiments.data_gen import *
from experiments.experiment import *
from evaluation_metric import *


results = []
embeddings = {}
tau = 0.5
edr = 0.2
fmr = 0.5
dim = 256
rad = [0.1, 0.2, 0.5, 1.0]

for name in ['Blob','Sphere','Circles','Moons','Swissroll','Scurve','Cora','Pubmed']:
    for i in np.arange(50):
        X, y_true, G = data_set(name, n_samples = 500, n_neighbours = 50,features = 'none', standardize = True, 
            centers = 4, cluster_std = [0.1,0.1,1.0,1.0],
            factor = 0.2, noise = 0.05,
            random_state = 0, radius = False, epsilon = 0.5, 
            SBMtype = 'lazy')
        new_data = convert_to_graph(X, n_neighbours = 50,features='none',standardize=True, epsilon = 0.5,eps = True)
        new_data.y = torch.from_numpy(y_true)
        for model_name in ['GRACE','DGI','BGRL','CCA-SSG']:
            for gnn_type in ['symmetric', 'RW']:
                for alpha in np.arange(0,1.1,0.1):
                    mod, res, out  = experiment(model_name, new_data,new_data.x,
                                new_data.y, None,
                                patience=20, epochs=200,
                                n_layers=2, out_dim=2, lr1=1e-3, lr2=1e-2, wd1=0.0,
                                wd2=0.0, tau=tau, lambd=1e-4, min_dist=0.1,
                                method='heat', n_neighbours=15,
                                norm='normalize', edr=edr, fmr=fmr,
                                proj="nonlinear", pred_hid=512, proj_hid_dim=dim,
                                dre1=0.2, dre2=0.2, drf1=0.4, drf2=0.4,
                                npoints = 500, n_neighbors = 50, classification = True, 
                                densmap = False, random_state = i, n = 15, perplexity = 30, 
                                alpha = alpha, beta = 1.0, gnn_type = gnn_type, 
                                name_file="blob-test",subsampling=None)
                    results += [res]
                    # out = mod.get_embedding(new_data)
                    embeddings[name + '_' + model_name + '_' + gnn_type + '_' + str(alpha)+'trial_'+str(i)]  =  {
                                                    'model': model_name, 
                                                    'alpha': alpha,
                                                    'gnn_type': gnn_type,   
                                                    'embedding' : out,
                                                    'alpha': alpha}
    file_path = os.getcwd() + '/' + name + '_results.csv'

    pd.DataFrame(np.array(results),
                    columns =['model', 'method',
                        'dim', 'neighbours', 'n_layers', 'norm','min_dist',
                            'dre1', 'drf1', 'lr', 'edr', 'fmr',
                        'tau', 'lambd','pred_hid','proj_hid_dim',
                        'sp','acc','local','density','alpha','beta','gnn_type']).to_csv(file_path)


    with open(os.getcwd() + name + '_results.pkl', 'wb') as file:
        # A new  file will be created
        pickle.dump(embeddings, file)

    print(results)