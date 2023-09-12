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


###TODO: n_clusters pipeline
###TODO: a,b
### Question: SIMPLE_GC_DEC
### TODO: mac_epochs


class SPAGCN(nn.Module):
    def __init__(self, in_dim=1000, nhid=512, n_clusters=10, alpha=0.5, out_dim=2, n_neighbors=15):
        super(SPAGCN, self).__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=nhid, out_dim=out_dim, n_layers=2, dropout_rate=0.2)
        self.mu = Parameter(torch.Tensor(n_clusters, out_dim))
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
        # print(loss)
        return loss

    def target_distribution(self, q):
        # weight = q ** 2 / q.sum(0)
        # return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, x, adj, lr=0.001, max_epochs=200, update_interval=5, weight_decay=5e-4, opt="sgd", init="kmeans",
            n_neighbors=10, res=0.4):
        loss_values = []
        self.trajectory = []
        print("Initializing cluster centers.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":  # faster, whichever
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features, _ = self.forward(x, adj)
        # long tensor error
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
            loss_values.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())
        return loss_values

    def predict(self, x, adj):
        z, q = self(torch.Tensor(x), torch.Tensor(adj))
        return z, q


"""
# Assuming spagcn_embedding contains the 3D points
spagcn_embedding_np = spagcn_embedding.detach().numpy()

# Extract x, y, and z coordinates
x_coords = spagcn_embedding_np[:, 0]
y_coords = spagcn_embedding_np[:, 1]
z_coords = spagcn_embedding_np[:, 2]

# Set up color mapping according to x_coords
colors = np.linspace(0, 1, len(x_coords))  # Assuming each point has a corresponding color value
cmap = plt.get_cmap('Spectral')  # Use the 'Spectral' colormap

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the scatter plot with color mapping
scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, cmap=cmap)

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add colorbar to show the color mapping
cbar = plt.colorbar(scatter)
cbar.set_label('Color Value')

# Set the plot title
plt.title('3D Scatter Plot of SpaGCN Embedding with Color Mapping')

plt.show()  # Display the plot
"""
########################################################
########################################################
