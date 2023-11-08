import matplotlib.pyplot as plt
import sys
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
from sklearn import manifold


class SPAGCN(nn.Module):
    def __init__(self,
                 in_dim=1000,
                 nhid=256,
                 alpha=0.5,
                 beta=0.5,
                 out_dim=2,
                 epochs=500):
        super().__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=nhid, out_dim=out_dim, n_layers=1, dropout_rate=0)
        self.alpha, self.beta, self.epochs, self.out_dim = alpha, beta, epochs, out_dim

    def forward(self, features, edge_index):
        """
        Updates current_embedding, calculates q (probability distribution of node connection in lowdim)
        """
        current_embedding = self.gc(features, edge_index)
        np_emb = current_embedding.detach().numpy()
        lowdim_dist = euclidean_distances(np_emb, np_emb)
        q = 1 / (1 + self.alpha * torch.pow(torch.tensor(lowdim_dist), (2*self.beta)))
        return current_embedding, q

    def loss_function(self, p, q):
        def CE(highd, lowd):
            # highd and lowd both have indim x indim dimensions
            highd, lowd = torch.tensor(highd), torch.tensor(lowd)
            eps = 1e-9 # To prevent log(0)
            return -torch.sum(highd * torch.log(lowd + eps) + (1 - highd) * torch.log(1 - lowd + eps))
        loss = CE(p, q)
        return loss

    def fit(self, features, sparse, edge_index, lr=0.005, opt='adam', weight_decay=0):
        loss_values = []
        print("Starting fit.")
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        """ 
        Probability distribution in highdim defined as the sparse adj matrix with weights.
        No further updates to p
        """
        p = torch.tensor(sparse)

        """ 
        Initial robability distribution in lowdim.
        q will be updated at each forward pass
        """
        lap_init = manifold.SpectralEmbedding(n_components=self.out_dim, n_neighbors=15)
        embeds_init = lap_init.fit_transform(features)
        lowdim_dist = euclidean_distances(embeds_init, embeds_init)
        q_initial = 1 / (1 + self.alpha * torch.pow(torch.tensor(lowdim_dist), (2 * self.beta)))

        self.train()
        for epoch in range(self.epochs):
            # TODO: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
            optimizer.zero_grad()
            if epoch == 1000:
                loss = self.loss_function(p, q_initial)
            else:
                current_embedding, q = self(features, edge_index)
                loss = self.loss_function(p, q)
            
            # loss_np = loss.item()
            # print("Epoch ", epoch, " |  Loss ", loss_np)
            # loss_values.append(loss_np)

            print(loss.type())
            loss.backward()
            optimizer.step()
        return loss_values

    def predict(self, features, edge_index):
        current_embedding, q = self(features, edge_index)
        return current_embedding, q