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
from sklearn.decomposition import PCA


class SPAGCN(nn.Module):
    def __init__(self,
                 in_dim=1000,
                 nhid=256,
                 alpha=0.5,
                 beta=0.5,
                 out_dim=2,
                 epochs=500):
        super().__init__()
        self.gc = GCN(in_dim=in_dim, hid_dim=nhid, out_dim=out_dim, n_layers=3, dropout_rate=0.2)
        self.alpha, self.beta, self.epochs, self.out_dim = alpha, beta, epochs, out_dim

    def forward(self, features, edge_index):
        """
        Updates current_embedding, calculates q (probability distribution of node connection in lowdim)
        """
        current_embedding = self.gc(features, edge_index)
        lowdim_dist = torch.cdist(current_embedding,current_embedding)
        q = 1 / (1 + self.alpha * torch.pow(lowdim_dist, (2*self.beta)))
        return current_embedding, q

    def loss_function(self, p, q):
        def CE(highd, lowd):
            # highd and lowd both have indim x indim dimensions
            #highd, lowd = torch.tensor(highd, requires_grad=True), torch.tensor(lowd, requires_grad=True)
            eps = 1e-9 # To prevent log(0)
            return -torch.sum(highd * torch.log(lowd + eps) + (1 - highd) * torch.log(1 - lowd + eps))
        loss = CE(p, q)
        return loss

    def fit(self, features, sparse, edge_index, lr=0.005, opt='adam', weight_decay=0, dens_lambda=2.0):
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
        rp1 = torch.multiply(p, torch.pow(torch.cdist(p,p),2))
        rp2 = torch.sum(p, axis=1) # sum edge weights over each row
        rp = torch.log(rp1/rp2 + 1e-9)
        
        """ 
        Initial robability distribution in lowdim.
        q will be updated at each forward pass
        """
        # TODO: original umap uses there own spectral initialization (pretty theoretical)
        # or pca, random, tcswspectral
        pca_init = PCA(n_components=2)
        embeds_init = pca_init.fit_transform(features)
        lowdim_dist = euclidean_distances(embeds_init, embeds_init)
        q_initial = 1 / (1 + self.alpha * torch.pow(torch.tensor(lowdim_dist), (2 * self.beta)))

        # lap_init = manifold.SpectralEmbedding(n_components=self.out_dim, n_neighbors=15)
        # embeds_init = lap_init.fit_transform(features)
        # lowdim_dist = euclidean_distances(embeds_init, embeds_init)
        # q_initial = 1 / (1 + self.alpha * torch.pow(torch.tensor(lowdim_dist), (2 * self.beta)))

        self.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            if epoch == 0:
                q = q_initial
                q.requires_grad_(True)
            else:
                _, q = self(features, edge_index)
            
            rq1 = torch.multiply(q, torch.pow(torch.cdist(q,q),2))
            rq2 = torch.sum(q, axis=1) # sum edge weights over each row
            rq = torch.log(rq1/rq2 + 1e-9)
            #print(rp.size(), rq.size()) # torch.Size([1000, 1000]) torch.Size([1000, 1000])
            # TODO: DENSITY corr = torch.cov(rp, rq) / torch.pow((torch.var(rp) * torch.var(rq)),0.5)
            loss = self.loss_function(p, q) #TODO- dens_lambda * corr
            loss_np = loss.item()
            print("Epoch ", epoch, " |  Loss ", loss_np)
            loss_values.append(loss_np)

            loss.backward()
            optimizer.step()
        return loss_values

    def predict(self, features, edge_index):
        current_embedding, q = self(features, edge_index)
        return current_embedding, q