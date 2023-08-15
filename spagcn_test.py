import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from experiments.simulation_utils import make_roll
import random

random.seed(12345)
N_NEIGHBOURS = 20
# target and u difference in simulation_utils.py?
# A: t and u create the ground truth
X, target, data, u = make_roll(c=0.6, v=4, omega=12, n_samples=2000, n_neighbours=N_NEIGHBOURS,
                               a=2, b=2, scale=0.5, plot=True, features=None,
                               standardize=True)
# To visualize
# x = X[:, 0]
# y = X[:, 1]
# z = X[:, 2]
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, z, c=target, cmap='Spectral')
# plt.show()

from models.baseline_models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected

# INPUT 1: sparse matrix, numpy array form
A_dist = kneighbors_graph(X, N_NEIGHBOURS, mode='distance', include_self=False) # adjacency matrix
edge_index, edge_weights = from_scipy_sparse_matrix(A_dist)
edge_index, edge_weights = to_undirected(edge_index, edge_weights)
# INPUT 2: input features: identity matrix
A = torch.eye(X.shape[0])

model = GCN(in_dim=2000, hid_dim=20, out_dim=3, n_layers=2, dropout_rate=0.2)
model(A, edge_index)

# QUESTION: n_clusters=1 ok? Also related to cluster_labels mentioned in Slack message
class GC_DEC(nn.Module):
    def __init__(self, in_dim, nhid=20, n_clusters=1, alpha=0.2):
        super(GC_DEC, self).__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=nhid, out_dim=3, n_layers=2, dropout_rate=0.2)
        # TODO: Need accurate hid_dim reflecting SpaGCN. Put 20 here for now.
        self.mu = Parameter(torch.Tensor(n_clusters, 3)) # for now
        self.n_clusters = n_clusters
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-6)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        # weight = q ** 2 / q.sum(0)
        # return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, x, adj, lr=0.001, max_epochs=10, update_interval=5, weight_decay=5e-4, opt="adam", init="kmeans",
            n_neighbors=10, res=0.4):
        self.trajectory = []
        print("Initializing cluster centers.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":  # faster, whichever
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(x,adj)
        #long tensor error
        # ----------------------------------------------------------------

        if init == "kmeans":
            # Uses umap-like functions
            # Kmeans only use exp info, no spatial
            # kmeans = KMeans(self.n_clusters, n_init=20)
            # y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
            # Kmeans use exp and spatial
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().numpy())
        elif init == "louvain":
            adata = sc.AnnData(features.detach().numpy())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
        # ----------------------------------------------------------------
        # x = torch.FloatTensor(x)
        # adj = torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(x, adj)
                p = self.target_distribution(q).data
            if epoch % 100 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(x, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

    def predict(self, x, adj):
        z, q = self(torch.Tensor(x), torch.Tensor(adj))
        return z, q

# QUESTION: dropout=0.5? Some models in train_models.py
model = GC_DEC(in_dim=A.shape[0])
model.fit(A, edge_index)
spagcn_embedding = model.predict(A, edge_index)[0]

########################################################
########################################################
"""
# from evaluation_metric import *
# def experiment(model_name, data, X,
#                target, device,
#                patience=20, epochs=500,
#                n_layers=2, out_dim=2, lr1=1e-3, lr2=1e-2, wd1=0.0,
#                wd2=0.0, tau=0.5, lambd=1e-4, min_dist=0.1,
#                method='heat', n_neighbours=15,
#                norm='normalize', edr=0.5, fmr=0.2,
#                proj="standard", pred_hid=512, proj_hid_dim=512,
#                dre1=0.2, dre2=0.2, drf1=0.4, drf2=0.4,
#                npoints=500, n_neighbors=50, classification=True,
#                densmap=False, random_state=42, n=15, perplexity=30,
#                alpha=0.5, beta=0.1, gnn_type='symmetric',
#                name_file="1", subsampling=None):
#     # num_classes = int(data.y.max().item()) + 1
#
#     if model_name == 'SpaGCN':
#         model = GC_DEC(n_components=2)
#         embeds = model.fit_transform(
#             X)  # StandardScaler().fit_transform(X) --already standardized when converting graphs
#
#     else:
#         raise ValueError("Model unknown!!")
#
#     sp, acc, local, density = eval_all(data, X, embeds, target, n_points=npoints, n_neighbors=n_neighbors,
#                                        classification=classification)
#     print("done with the embedding evaluation")
#
#     results = [model_name,
#                method,
#                out_dim,
#                n_neighbors,
#                n_layers,
#                norm,
#                min_dist,
#                dre1,
#                drf1,
#                lr1,
#                edr,
#                fmr,
#                tau,
#                lambd,
#                pred_hid,
#                proj_hid_dim,
#                sp,
#                acc,
#                local,
#                density,
#                alpha,
#                beta,
#                gnn_type
#                ]
#
#     return (model, results, embeds)
"""
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")