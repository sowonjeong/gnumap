temp_model_name = "UMAP"
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
from spagcn_test import *
from experiments.create_dataset import *

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

n_samples = 1000
X_ambient, X_manifold, cluster_labels, G = create_dataset(args.name, n_samples=n_samples,
                                                          features='none', standardize=True,
                                                          centers=4, cluster_std=[0.1, 0.1, 1.0, 1.0],
                                                          ratio_circles=0.2, noise=args.noise, random_state=12345,
                                                          radius_knn=args.radius_knn, bw=args.bw,
                                                          SBMtype='lazy',
                                                          a=args.a,
                                                          b=args.b)


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

def experiment(model_name, G, X_ambient, X_manifold,
               cluster_labels,
               patience=20, epochs=500,
               n_layers=2, out_dim=2, hid_dim=16, lr=1e-3, wd=0.0,
               tau=0.5, lambd=1e-4, min_dist=1e-3, edr=0.5, fmr=0.2,
               proj="standard", pred_hid=12,
               n_neighbors=15, dataset="Blobs",
               random_state=42, perplexity=30,
               alpha=0.5, beta=0.1, gnn_type='symmetric',
               name_file="1"):
    # num_classes = int(data.y.max().item()) + 1
    if model_name == 'DGI':
        model = train_dgi(G, hid_dim=hid_dim, out_dim=out_dim,
                          n_layers=n_layers,
                          patience=patience,
                          epochs=epochs, lr=lr,
                          name_file=name_file,
                          alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)

    elif model_name == 'GRACE':
        model, loss = train_grace(G, channels=hid_dim, proj_hid_dim=out_dim,
                                  tau=tau,
                                  epochs=epochs, lr=lr, wd=wd,
                                  fmr=fmr, edr=edr, proj=proj, name_file=name_file,
                                  alpha=alpha, beta=beta, gnn_type=gnn_type)
        embeds = model.get_embedding(G)

    elif model_name == 'CCA-SSG':
        model = train_cca_ssg(G, hid_dim=hid_dim,
                              channels=out_dim,
                              lambd=lambd,
                              n_layers=n_layers,
                              epochs=epochs, lr=lr,
                              fmr=fmr, edr=edr, name_file=name_file)
        embeds = model.get_embedding(G)
    elif model_name == 'Entropy-SSG':
        model = train_entropy_ssg(G, hid_dim=hid_dim,
                                  channels=out_dim,
                                  lambd=lambd,
                                  n_layers=n_layers,
                                  epochs=epochs, lr=lr,
                                  fmr=fmr, edr=edr, name_file=name_file)
        embeds = model.get_embedding(G)

    elif model_name == 'BGRL':
        model = train_bgrl(G, hid_dim, out_dim,
                           lambd=lambd,
                           n_layers=n_layers,
                           epochs=epochs, lr=lr,
                           fmr=fmr, edr=edr,
                           pred_hid=pred_hid, wd=wd,
                           drf1=fmr, drf2=fmr, dre1=edr,
                           dre2=edr, name_file=name_file)
        embeds = model.get_embedding(G)
    elif model_name == "GNUMAP":
        raise ValueError("Not implemented yet!!")
    elif model_name == "SpaGCN":
        edge_index = G.edge_index
        A = torch.eye(n_samples)  # identity feature matrix
        model = GC_DEC(in_dim=n_samples, out_dim=out_dim)
        model.fit(A, edge_index)
        embeds = model.predict(A, edge_index)[0]
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
    # QUESTION: A lot of penalty
    # embeds.detach().numpy() for spagcn
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

    """
    keys = results.keys()
    values = results.values()

    # Check if the file exists
    file_exists = os.path.isfile('data.csv')

    # Set mode based on file existence
    mode = 'a' if file_exists else 'w'

    # Write the data to the CSV file
    with open('data.csv', mode=mode, newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(keys)
        writer.writerow(values)
        """

    return (model, results, embeds)


for i in range(5):
    model, results, embeds = experiment(temp_model_name, G, X_ambient, X_manifold, cluster_labels,
                                        patience=20, epochs=args.epoch,
                                        n_layers=args.n_layers, out_dim=X_manifold.shape[1],
                                        hid_dim=args.hid_dim, lr=args.lr, wd=0,
                                        tau=np.nan, lambd=np.nan, alpha=0.5, beta=1,
                                        gnn_type='symmetric', dataset=args.name)

plt.figure()
#embeds = embeds.detach().numpy()
print(embeds)
plt.scatter(*embeds.T, s=10, c=cluster_labels, alpha=0.5, cmap='Spectral')
plt.title(temp_model_name)
plt.show()
