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
import sklearn.manifold as manifold
from sklearn.decomposition import PCA
import umap

sys.path.append('../')
from data_utils import *
from edge_prediction import edge_prediction
from graph_utils import get_weights, transform_edge_weights
from label_classification import label_classification
from models.baseline_models import *
from models.train_models import *
from evaluation_metric import *
from train_utils import *
from umap_functions import *


def experiment(model_name, data,X,
               target, device,
               patience=20, epochs=500,
               n_layers=2, out_dim=2, lr1=1e-3, lr2=1e-2, wd1=0.0,
               wd2=0.0, tau=0.5, lambd=1e-4, min_dist=0.1,
               method='heat', n_neighbours=15,
               norm='normalize', edr=0.5, fmr=0.2,
               proj="standard", pred_hid=512, proj_hid_dim=512,
               dre1=0.2, dre2=0.2, drf1=0.4, drf2=0.4,
               npoints = 500, n_neighbors = 5, classification = True, 
               densmap = False, random_state = 42, n = 15, perplexity = 30, 
               alpha = 0.5, beta = 0.1, gnn_type = 'symmetric', 
               name_file="1",subsampling=None):

    num_classes = int(data.y.max().item()) + 1
    if model_name == 'DGI':
        model = train_dgi(data, hid_dim=proj_hid_dim, out_dim=out_dim,
                          n_layers=n_layers,
                          patience=patience,
                          epochs=epochs, lr=lr1,
                          name_file=name_file,
                          alpha = alpha, beta = beta, gnn_type = gnn_type)
        embeds = model.get_embedding(data)

    elif model_name == 'GRACE':
        model, loss =  train_grace(data, channels=proj_hid_dim, proj_hid_dim=out_dim,
                             tau=tau,
                             epochs=epochs, lr=lr1, wd=wd1,
                             fmr=fmr, edr=edr, proj=proj, name_file=name_file,
                             alpha = alpha, beta = beta, gnn_type = gnn_type)
        embeds = model.get_embedding(data)

    elif model_name == 'CCA-SSG':
        model, loss =  train_cca_ssg(data,proj_hid_dim, out_dim, 
                               lambd=lambd,
                               n_layers=n_layers,
                               epochs=epochs, lr=lr1,
                               fmr=fmr, edr=edr, name_file=name_file,
                               alpha = alpha, beta = beta, gnn_type = gnn_type)
        embeds = model.get_embedding(data)

    elif model_name == 'BGRL':
        model =  train_bgrl(data, proj_hid_dim, out_dim, 
                               lambd=lambd,
                               n_layers=n_layers,
                               epochs=epochs, lr=lr1,
                               fmr=fmr, edr=edr,
                               pred_hid=pred_hid,  wd=wd1,
                               drf1=drf1, drf2=drf2, dre1=dre1,
                               dre2=dre2,name_file=name_file,
                               alpha = alpha, beta = beta, gnn_type = gnn_type)
        embeds = model.get_embedding(data)
    
    elif model_name == 'PCA':
        model =  PCA(n_components = 2)
        embeds = model.fit_transform(StandardScaler().fit_transform(X))
    elif model_name == 'LaplacianEigenmap':
        model = manifold.SpectralEmbedding(n_components = 2,n_neighbors = 5)
        embeds = model.fit_transform(StandardScaler().fit_transform(X))
    elif model_name == 'Isomap':
        model = manifold.Isomap(n_components = 2)
        embeds = model.fit_transform(StandardScaler().fit_transform(X))
    elif model_name == 'TSNE':
        model = manifold.TSNE(n_components = 2, learning_rate  = 'auto', init = 'random', perplexity = perplexity)
        embeds = model.fit_transform(X)
    elif model_name == 'UMAP':
        model = umap.UMAP(random_state=random_state, n_neighbors = n, min_dist = min_dist)
        embeds = model.fit_transform(X)
    elif model_name == 'DenseMAP':
        model = umap.UMAP(random_state=random_state, densmap = True, n_neighbors = n, min_dist = min_dist)
        embeds = model.fit_transform(X)
    else:
        raise ValueError("Model unknown!!")


    sp, acc, local, density = eval_all(data, data.x, embeds, data.y, n_points = npoints, n_neighbors = n_neighbors, classification = classification)
    print("done with the embedding evaluation")


    results = [  model_name,
                 method,
                 out_dim,
                 n_neighbors,
                 n_layers,
                 norm,
                 min_dist,
                 dre1,
                 drf1,
                 lr1,
                 edr,
                 fmr,
                 tau,
                 lambd,
                 pred_hid,
                 proj_hid_dim,
                 sp,
                 acc,
                 local,
                 density,
                 alpha,
                 beta,
                 gnn_type
    ]

    return(model, results)
