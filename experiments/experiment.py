import argparse
from carbontracker.tracker import CarbonTracker
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
from metrics.edge_prediction import edge_prediction
from graph_utils import get_weights, transform_edge_weights
from metrics.label_classification import label_classification
from models.baseline_models import *
from models.train_models import *
from metrics.evaluation_metrics import *
from gnumap.umap_functions import *


def experiment(model_name, data, X_ambient, X_manifold, 
               cluster_labels,
               patience=20, epochs=500,
               n_layers=2, out_dim=2, hid_dim=16, lr=1e-3, wd=0.0,
               tau=0.5, lambd=1e-4, min_dist=1e-3, edr=0.5, fmr=0.2,
               proj="standard", pred_hid=512, 
               npoints = 500, n_neighbors = 15, classification = True,
               random_state = 42, perplexity = 30, 
               alpha = 0.5, beta = 0.1, gnn_type = 'symmetric', 
               name_file="1", subsampling=None):
   
    # num_classes = int(data.y.max().item()) + 1
    if model_name == 'DGI':
        model = train_dgi(data, hid_dim=hid_dim, out_dim=out_dim,
                          n_layers=n_layers,
                          patience=patience,
                          epochs=epochs, lr=lr,
                          name_file=name_file,
                          alpha = alpha, beta = beta, gnn_type = gnn_type)
        embeds = model.get_embedding(data)

    elif model_name == 'GRACE':
        model, loss =  train_grace(data, channels=hid_dim, proj_hid_dim=out_dim,
                             tau=tau,
                             epochs=epochs, lr=lr, wd=wd,
                             fmr=fmr, edr=edr, proj=proj, name_file=name_file,
                             alpha = alpha, beta = beta, gnn_type = gnn_type)
        embeds = model.get_embedding(data)

    elif model_name == 'CCA-SSG':
        model =  train_cca_ssg(data, hid_dim=hid_dim,
                               channels=out_dim,
                               lambd=lambd,
                               n_layers=n_layers,
                               epochs=epochs, lr=lr,
                               fmr=fmr, edr=edr, name_file=name_file)
        embeds = model.get_embedding(data)
    elif model_name == 'Entropy-SSG':
        model =  train_entropy_ssg(data, hid_dim=hid_dim,
                               channels=out_dim,
                               lambd=lambd,
                               n_layers=n_layers,
                               epochs=epochs, lr=lr,
                               fmr=fmr, edr=edr, name_file=name_file)
        embeds = model.get_embedding(data)

    elif model_name == 'BGRL':
        model =  train_bgrl(data, hid_dim, out_dim, 
                            lambd=lambd,
                            n_layers=n_layers,
                            epochs=epochs, lr=lr,
                            fmr=fmr, edr=edr,
                            pred_hid=pred_hid,  wd=wd,
                            drf1=fmr, drf2=fmr, dre1=edr,
                            dre2=edr,name_file=name_file)
        embeds = model.get_embedding(data)
    elif model_name == "GNUMAP":
        raise ValueError("Not implemented yet!!")   
    elif model_name == "SpaGCN":
        raise ValueError("Not implemented yet!!")
    elif model_name == 'PCA':
        model =  PCA(n_components = 2)
        embeds = model.fit_transform(X_ambient) # StandardScaler().fit_transform(X) --already standardized when converting graphs
    elif model_name == 'LaplacianEigenmap':
        model = manifold.SpectralEmbedding(n_components = 2,n_neighbors = n_neighbors)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'Isomap':
        model = manifold.Isomap(n_components = 2)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'TSNE':
        model = manifold.TSNE(n_components = 2, learning_rate  = 'auto', 
                              init = 'random', perplexity = perplexity)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'UMAP':
        model = umap.UMAP(n_components = 2, random_state=random_state, 
                          n_neighbors = n_neighbors, min_dist = min_dist)
        embeds = model.fit_transform(X_ambient)

    elif model_name == 'DenseMAP':
        model = umap.UMAP(n_components = 2, random_state=random_state, 
                          densmap = True, n_neighbors = n_neighbors, 
                          min_dist = min_dist)
        embeds = model.fit_transform(X_ambient)
    else:
        raise ValueError("Model unknown!!")

    global_metrics, local_metrics = eval_all(G, X_ambient, X_manifold, embeds, cluster_labels,  dataset = "Blobs")
    print("done with the embedding evaluation")
    
    results = {**global_metrics, **local_metrics}
    results['model_name'] = model_name
    results['out_dim'] = out_dim
    results['hid_dim'] = hid_dim
    results['n_neighbors'] = n_neighbors
    results['min_dist'] = min_dist
    results['lr'] = lr
    results['edr'] = edr
    results['fmr'] = fmr
    results['tau'] = tau
    results['lambd'] = lambd
    results['pred_hid'] = pred_hid
    results['alpha_gnn'] = alpha
    results['beta_gnn'] = beta
    results['gnn_type'] = gnn_type

    return(model, results, embeds)
