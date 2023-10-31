import matplotlib.pyplot as plt
import sys

# target and u difference in simulation_utils.py?
# A: t and u create the ground truth

sys.path.append('../')
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


# # INPUT 1: sparse matrix, numpy array form
# A_dist = kneighbors_graph(X, N_NEIGHBOURS, mode='distance', include_self=False) # adjacency matrix
# edge_index, edge_weights = from_scipy_sparse_matrix(A_dist)
# edge_index, edge_weights = to_undirected(edge_index, edge_weights)

# # INPUT 2: input features: identity matrix
# A = torch.eye(X.shape[0])


class SPAGCN(nn.Module):
    def __init__(self,
                 in_dim=1000,
                 nhid=256,
                 n_clusters=10,  # kmeans
                 alpha=0.5,
                 out_dim=2,
                 n_neighbors=15):  # louvain
        super(SPAGCN, self).__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=nhid, out_dim=out_dim, n_layers=1, dropout_rate=0)
        self.mu = Parameter(torch.Tensor(n_clusters, out_dim))
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, x, adj,
            lr=0.005,
            max_epochs=100,
            update_interval=3,
            weight_decay=0,
            opt="adam",
            init="kmeans",
            n_neighbors=10,
            res=0.4,
            init_spa=True,
            tol=1e-3):
        loss_values = []
        print("Starting fit.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(x, adj)

        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean()) # mu determined by initial fitting

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(x, adj)
                p = self.target_distribution(q).data
            if epoch % 50 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(x, adj)
            loss = self.loss_function(p, q)
            loss_values.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
        return loss_values

    def predict(self, x, adj):
        z, q = self(torch.Tensor(x), torch.Tensor(adj))
        return z, q