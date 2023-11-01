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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import SpectralEmbedding


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
        self.alpha = alpha
        self.embeds = Parameter(torch.Tensor(in_dim, out_dim))

    def prob_high_dim(self, sigma, dist_row):
        """ For each row in dist, compute prob in high dim (1d array)"""
        d = dist[dist_row] - rho[dist_row] # list of all dist in the row - mindist
        d[d<0] = 0 # to avoid float errors
        return np.exp(-d/sigma)    
    
    def k(self, prob): # gets number of nearest value
        return np.power(2, np.sum(prob))

    def prob_low_dim(self, Y):
        a=1
        b=1
        inv_distances = np.power(1+a*np.square(euclidean_distances(Y,Y))**b, -1)
        return inv_distances

    def sigma_binary_search(self, k_of_sigma, fixed_k):
        """
        Solve equation k_of_sigma(sigma) = fixed_k 
        with respect to sigma by the binary search algorithm
        """
        sigma_lower_limit = 0; sigma_upper_limit = 1000
        for i in range(20):
            approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
            if k_of_sigma(approx_sigma) < fixed_k:
                sigma_lower_limit = approx_sigma
            else:
                sigma_upper_limit = approx_sigma
            if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
                break
        return approx_sigma

    def forward(self, x, adj):
        z = self.gc(x, adj)
        q = prob_low_dim(z)
        return z, q

    def loss_function(self, p, q):
        def ce(p,q):
            return -p * np.log(q+0.01) - (1-p) * np.log(1-a + 0.01)
        loss = ce(p,q)
        return loss

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

        # -------------- p is calculated here - almost identical to umap high dimension probability
        features = self.gc(x, adj)
        features = features.detach().numpy()
        dist = np.square(euclidean_distances(features,features))
        rho = [sorted(dist[i])[1] for i in range(dist.shape[0])] # min dist for each pt
        N_NEIGHBOR = 15
        prob = np.zeros((x.shape[0],x.shape[0]))
        sigma_array = []
        for dist_row in range(x.shape[0]):
            func = lambda sigma: k(self.prob_high_dim(sigma, dist_row))
            binary_search_result = self.sigma_binary_search(func, N_NEIGHBOR)
            prob[dist_row] = self.prob_high_dim(binary_search_result, dist_row)
            sigma_array.append(binary_search_result)
            if (dist_row + 1) % 100 == 0:
                print("Sigma binary search finished {0} of {1} rows".format(dist_row + 1, n))
        print("\nMean sigma = " + str(np.mean(sigma_array)))
        p = (prob + np.transpose(prob)) / 2
        # ------------------------------------------------------------------------------------

        self.train()
        for epoch in range(max_epochs):
            if epoch == 0:
                # initialize q
                init_model = SpectralEmbedding(n_components = 2, n_neihbors=50)
                y = init_model.fit_trainsform(x)
                q = self.prob_low_dim(y)
            print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(x, adj)
            loss = self.loss_function(p, q)
            loss_values.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
        return loss_values

    def predict(self, x, adj):
        return z, q