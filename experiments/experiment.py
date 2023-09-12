import csv
import argparse
from carbontracker.tracker import CarbonTracker
import copy, collections
import networkx as nx, numpy as np
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
from models.spagcn import *
from experiments.create_dataset import *


def visualize_dataset(X, cluster_labels, title, file_name):
    fig = plt.figure()
    ax = fig.gca()

    if X is not None:
        ax.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap=plt.cm.Spectral)
        ax.set_title(title)
    else:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

    save_path = os.path.join(os.getcwd(), 'results', file_name)
    plt.savefig(save_path, format='png', dpi=300, facecolor=fig.get_facecolor())
    plt.close()


def experiment(model_name, G, X_ambient, X_manifold,
               cluster_labels,
               patience=20, epochs=500,
               n_layers=2, out_dim=2, hid_dim=128, lr=1e-4, wd=0.0,
               tau=0.5, lambd=1e-4, min_dist=1e-3, edr=0.8, fmr=0,
               proj="standard", pred_hid=512,
               n_neighbors=15, dataset='Blobs',
               random_state=42, perplexity=30,
               alpha=0.5, beta=0.1, gnn_type='symmetric',
               name_file="1", save_img=False):
    # num_classes = int(data.y.max().item()) + 1
    loss_values = [1]

    if model_name == 'DGI':  # a b type
        model, loss_values = train_dgi(G, hid_dim=hid_dim, out_dim=out_dim,
                                       n_layers=n_layers,
                                       patience=patience,
                                       epochs=epochs, lr=1e-4,
                                       name_file=name_file,
                                       alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)

    elif model_name == 'GRACE':  # a b type t
        model, loss_values = train_grace(G, channels=hid_dim, proj_hid_dim=out_dim,
                                         tau=tau,
                                         epochs=100, lr=1e-4, wd=wd,
                                         fmr=fmr, edr=edr, proj=proj, name_file=name_file,
                                         alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)

    elif model_name == 'CCA-SSG':  # a b type lam
        model, loss_values = train_cca_ssg(G, hid_dim=hid_dim,
                                           channels=out_dim,
                                           lambd=lambd,
                                           n_layers=n_layers,
                                           epochs=100, lr=1e-4,
                                           fmr=fmr, edr=edr, name_file=name_file)
        embeds = model.get_embedding(G)
    elif model_name == 'Entropy-SSG':
        model, loss_values = train_entropy_ssg(G, hid_dim=hid_dim,
                                               channels=out_dim,
                                               lambd=lambd,
                                               n_layers=n_layers,
                                               epochs=epochs, lr=lr,
                                               fmr=fmr, edr=edr, name_file=name_file)
        embeds = model.get_embedding(G)

    elif model_name == 'BGRL':  # lamb a b type
        model, loss_values = train_bgrl(G, hid_dim, out_dim,
                                        lambd=lambd,
                                        n_layers=n_layers,
                                        epochs=epochs, lr=2e-4,
                                        fmr=fmr, edr=edr,
                                        pred_hid=pred_hid, wd=wd,
                                        drf1=fmr, drf2=fmr, dre1=edr,
                                        dre2=edr, name_file=name_file)
        embeds = model.get_embedding(G)
    elif model_name == "GNUMAP":  # alpha beta type
        model, embeds, loss_values = train_gnumap(G, hid_dim, out_dim,
                                                  n_layers=n_layers,
                                                  epochs=epochs, lr=1e-4, wd=wd, name_file=name_file)
    elif model_name == "SPAGCN":  # alpha
        edge_index = G.edge_index
        A = torch.eye(X_ambient.shape[0])  # identity feature matrix
        model = SPAGCN(in_dim=X_ambient.shape[0], out_dim=out_dim, n_neighbors=n_neighbors)
        loss_values = model.fit(A, edge_index)
        embeds = model.predict(A, edge_index)[0]
        embeds = embeds.detach().numpy()
    elif model_name == 'PCA':
        model = PCA(n_components=2)
        embeds = model.fit_transform(
            X_ambient)  # StandardScaler().fit_transform(X) --already standardized when converting graphs
    elif model_name == 'LaplacianEigenmap':
        model = manifold.SpectralEmbedding(n_components=2, n_neighbors=n_neighbors)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'Isomap':
        model = manifold.Isomap(n_components=2)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'TSNE':
        model = manifold.TSNE(n_components=2, learning_rate='auto',
                              init='random', perplexity=perplexity)
        embeds = model.fit_transform(X_ambient)
    elif model_name == 'UMAP':
        model = umap.UMAP(n_components=2, random_state=random_state,
                          n_neighbors=n_neighbors, min_dist=min_dist)
        embeds = model.fit_transform(X_ambient)

    elif model_name == 'DenseMAP':
        model = umap.UMAP(n_components=2, random_state=random_state,
                          densmap=True, n_neighbors=n_neighbors,
                          min_dist=min_dist)
        embeds = model.fit_transform(X_ambient)
    else:
        raise ValueError("Model unknown!!")

    file_name = (
        f"{model_name}_{dataset}_tau_{tau}_lambda_{lambd}_gnn_type_{gnn_type}_"
        f"alpha_{alpha}_beta_{beta}_{name_file}.png"
    )
    if save_img:
        visualize_dataset(embeds, cluster_labels, title=dataset, file_name=file_name)
    else:
        pass

    try:
        loss_values = [item.item() for item in loss_values]
    except:
        pass

    if np.isnan(loss_values[-1]):
        embeds = None
        results = None
    else:
        global_metrics, local_metrics = eval_all(G, X_ambient, X_manifold, embeds, cluster_labels,
                                                 dataset=dataset)
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

    return (model, results, embeds, loss_values)
