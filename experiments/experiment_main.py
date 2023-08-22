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
# from carbontracker.tracker import CarbonTracker
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
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from sklearn.datasets import *

sys.path.append('../')
from models.baseline_models import *
from models.train_models import *
from gnumap.umap_functions import *
from graph_utils import *
from experiments.create_dataset import *
from experiments.experiment import *
from metrics.evaluation_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Blobs')
parser.add_argument('--filename', type=str, default='test')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--hid_dim', type=int, default=2)  # 512
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--a', type=float, default=1.)  # data construction
parser.add_argument('--b', type=float, default=1.)  # data construction
parser.add_argument('--radius_knn', type=float, default=0.1)  # graph construction
parser.add_argument('--bw', type=float, default=1.)  # graph construction
args = parser.parse_args()


def get_next_file_path(base_path):
    """
    Returns the next available file path by appending an index to the base path.
    For example, if "results.csv" exists, it will return "results_0.csv".
    If "results_0.csv" also exists, it will return "results_1.csv", and so on.
    """
    counter = 0
    # Split the base_path into name and extension
    name, ext = os.path.splitext(base_path)

    # The first file path to try is the base_path itself
    next_path = base_path

    while os.path.exists(next_path):
        next_path = f"{name}_{counter}{ext}"
        counter += 1

    return next_path


# seed = 12345
# np.random.seed(seed)
# torch.manual_seed(seed)

name = args.name

results = {}
file_path = os.getcwd() + '/results/' + name + '_' + str(args.radius_knn) + '_gnn_results.csv'

X_ambient, X_manifold, cluster_labels, G = create_dataset(args.name, n_samples=1000,
                                                          features='none', standardize=True,
                                                          centers=4, cluster_std=[0.1, 0.1, 1.0, 1.0],
                                                          ratio_circles=0.2, noise=args.noise,
                                                          radius_knn=args.radius_knn, bw=args.bw,
                                                          SBMtype='lazy',
                                                          a=args.a,
                                                          b=args.b)
import matplotlib.pyplot as plt

def visualize_dataset(X_ambient, cluster_labels, title, save_path=True):
    plt.scatter(X_ambient[:, 0], X_ambient[:, 1], c=cluster_labels, cmap=plt.cm.Spectral)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()


visualize_dataset(X_ambient, cluster_labels, title=name, save_path=os.getcwd() + '/results/' + "gt_" +name+ ".png")

for model_name in ['DGI', 'BGRL']:
    mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                               patience=20, epochs=args.epoch,
                               n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                               hid_dim=args.hid_dim, lr=args.lr, wd=0,
                               tau=np.nan, lambd=np.nan, alpha=0.5, beta=1,
                               gnn_type='symmetric', dataset=args.name)
    visualize_dataset(out, cluster_labels, title=name, save_path=os.getcwd() + '/results/' + model_name+name+ ".png")
    results[name + '_' + model_name] = res

# 'GRACE', 'GNUMAP', 'CCA-SSG', 'SPAGCN'
for model_name in ['SPAGCN']:
    for gnn_type in ['symmetric', 'RW']:
        for alpha in np.arange(0, 1.1, 0.1):
            if model_name == "GRACE":  # Loss nan
                for tau in [0.01, 0.1, 0.2, 0.5, 1., 10]:
                    mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                               patience=20, epochs=args.epoch,
                                               n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                               hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                               tau=tau, lambd=1e-4, min_dist=1e-3, edr=0.2, fmr=0.2,
                                               proj="standard", pred_hid=args.hid_dim, n_neighbors=15,
                                               random_state=42, perplexity=30, alpha=alpha, beta=1,
                                               gnn_type=gnn_type,
                                               name_file="logsGRACE " + name, dataset=args.name)
                    results[name + '_' + model_name + '_' + gnn_type + '_' + str(alpha) + '_tau_' + str(tau)] = res
            elif model_name == 'CCA-SSG':  # tau pipeline issue
                for lambd in [1e-3, 5 * 1e-2, 1e-2, 5 * 1e-1, 1e-1, 1.]:
                    mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                               patience=20, epochs=args.epoch,
                                               n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                               hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                               tau=tau, lambd=1e-4, min_dist=1e-3, edr=0.2, fmr=0.2,
                                               proj="standard", pred_hid=args.hid_dim,
                                               n_neighbors=15,
                                               random_state=42, perplexity=30, alpha=alpha, beta=1,
                                               gnn_type=gnn_type,
                                               name_file="logsCCA-SSG " + name, dataset=args.name)
                    results[name + '_' + model_name + '_' + gnn_type + '_' + str(alpha) + '_lambd_' + str(lambd)] = res
            elif model_name == 'SPAGCN':
                lambd = 1e-4
                model, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                             patience=20, epochs=args.epoch,
                                             n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                             hid_dim=args.hid_dim, lr=args.lr, wd=0,
                                             tau=np.nan, lambd=lambd, alpha=alpha, beta=1,
                                             gnn_type=gnn_type, dataset=args.name)
                visualize_dataset(out, cluster_labels, title=name,
                                  save_path=os.getcwd() + '/results/' + model_name + name + gnn_type + str(alpha) +".png")
                results[name + '_' + model_name + '_' + gnn_type + '_' + str(alpha) + '_lambd_' + str(lambd)] = res
            else:
                lambd = 1e-4
                mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                                           patience=20, epochs=args.epoch,
                                           n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                           hid_dim=args.hid_dim, lr=args.lr, wd=0.0,
                                           tau=tau, lambd=lambd, min_dist=1e-3, edr=0.2, fmr=0.2,
                                           proj="standard", pred_hid=args.hid_dim, n_neighbors=np.nan,
                                           random_state=42, perplexity=30, alpha=alpha, beta=1,
                                           gnn_type=gnn_type,
                                           name_file="logsGNUMAP " + name, dataset=args.name)
                results[name + '_' + model_name + '_' + gnn_type + '_' + str(alpha) + '_lambd_' + str(lambd)] = res
# 'UMAP', 'DenseMAP' error - left out for now
for model_name in ['PCA', 'LaplacianEigenmap', 'Isomap', 'TSNE']:
    mod, res, out = experiment(model_name, G, X_ambient, X_manifold, cluster_labels,
                               out_dim=X_manifold.shape[1], dataset=args.name)
    visualize_dataset(out, cluster_labels, title=name,
                      save_path=os.getcwd() + '/results/' + model_name + name + ".png")
    results[name + '_' + model_name] = res

pd.DataFrame.from_dict(results, orient='index').to_csv(get_next_file_path(file_path))

# To visualize 3d
# x = X_ambient[:, 0]
# y = X_ambient[:, 1]
# z = X_ambient[:, 2]
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, z, c=target, cmap='Spectral')
# plt.show()

# 2d doesn't work
# x = X_ambient[:, 0]
# y = X_ambient[:, 1]
# plt.scatter(x, y, c=target, cmap='Spectral')

# Embedding
# plt.figure()
# embeds = embeds.detach().numpy()
# print(embeds)
# plt.scatter(*embeds.T, s=10, c=cluster_labels, alpha=0.5, cmap='Spectral')
# plt.title(temp_model_name)
# plt.show()
