import numpy as np
import scipy as sc
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected



def assign_label(point, xedges, yedges, labels):
    # Find the bin each point belongs to.
    i = np.digitize(point[0], xedges) - 1
    j = np.digitize(point[1], yedges) - 1
    # Return the label associated with that bin.
    return labels[i, j]


def make_roll(c=0.6, v=4, omega=12, n_samples = 2000, n_neighbours = 30,
              a = 2, b = 2, scale=0.5, plot=True, features=None,
              standardize=True):

    t = np.random.beta(a=b, b=b, size=n_samples)
    u = np.random.beta(a=2, b=2, size=n_samples) 
    vals = sc.stats.beta.pdf(t, a, b)
    x =4 *(v * t+c)* np.cos(omega  *t) + np.random.normal(scale=scale, size=n_samples)
    y =4 *(v * t+c) * np.sin(omega  *t) + np.random.normal(scale=scale, size=n_samples)
    z = 2 * np.ceil(np.max(x)) * (u-.5)

    X = np.vstack([np.array(x), np.array(y), np.array(z)]).T
    A = kneighbors_graph(X, n_neighbours, mode='distance', include_self=False) # edge weight is given by distance
    edge_index, edge_weights = from_scipy_sparse_matrix(A)
    edge_index, edge_weights = to_undirected(edge_index, edge_weights)
    M = torch.max(edge_weights)
    if standardize:
        preproc = StandardScaler()
        X = preproc.fit_transform(X)
    if features == 'coordinates':
        new_data = Data(x=torch.from_numpy(X).float(),
                        edge_index=edge_index,
                        edge_weight=edge_weights/M)
    else:
        new_data = Data(x=torch.eye(n_samples), edge_index=edge_index,
                        edge_weight=edge_weights/M)
    return(X, t, new_data, u)


def create_sphere(r, size = 1000, a = 3, b=0.5, noise= 0,
                  n_bins = 10):
    '''
    create sphere with center (0,0,0) and radius r
    '''
    theta = 2* np.pi * np.random.beta(a=a, b=b, size = size)
    phi = np.random.uniform(0, np.pi,size = size)

    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    
    if noise > 0 :
        x = x + np.random.normal(0, noise, size = size )
        y = y + np.random.normal(0, noise, size = size )
        z = z + np.random.normal(0, noise, size = size )
    X_manifold = np.stack([np.cos(theta), np.cos(phi)]).T
    hist, xedges, yedges = np.histogram2d(X_manifold[:, 0], X_manifold[:, 1], bins=(n_bins, n_bins))
    labels = np.arange(0, (n_bins + 1) * (n_bins + 1) ).reshape((n_bins + 1, n_bins + 1))
    cluster_labels = [assign_label(X_manifold[i,:], xedges, yedges, labels) for i in range(X_manifold.shape[0])]
    return np.stack([x,y,z]).T, X_manifold, np.array(cluster_labels)

def create_circles(ratio = 0.3, size = 1000, a = 1, b=1, noise= 0,
                  n_bins = 10):
    '''
    create circles
    '''
    theta = 2* np.pi * np.random.beta(a=a, b=b, size = size)
    z = np.random.binomial(1, 0.5, size=size)
    r = np.array([zz  + (1-zz) * ratio for zz in z])
    
    x = np.multiply(r, np.cos(theta))
    y = np.multiply(r, np.sin(theta))
    
    
    if noise > 0 :
        x = x + np.random.normal(0, noise, size = size )
        y = y + np.random.normal(0, noise, size = size )
    X_manifold = np.stack([np.cos(theta), z]).T
    hist, xedges, yedges = np.histogram2d(X_manifold[:, 0], X_manifold[:, 1], bins=(n_bins, n_bins))
    labels = np.arange(0, (n_bins + 1) * (n_bins + 1) ).reshape((n_bins + 1, n_bins + 1))
    cluster_labels = [assign_label(X_manifold[i,:], xedges, yedges, labels) for i in range(X_manifold.shape[0])]
    return np.stack([x,y,z]).T, X_manifold, np.array(cluster_labels)


def create_moons(size = 1000, a = 1, b=1, noise= 0,
                  n_bins = 10):
    '''
    create moons
    '''
    angle = np.pi * np.random.beta(a=a, b=b, size = size)
    z = np.random.binomial(1, 0.5, size=size)
    mean = np.array([zz * 0.5  - (1-zz) * 0.5 for zz in z])
    mean_y = np.array([zz * 0  + (1-zz) * 0.3 for zz in z])
    theta = np.pi * z - angle 
    x = mean +   np.cos(theta)
    y = mean_y + np.sin(theta)
    
    if noise > 0 :
        x = x + np.random.normal(0, noise, size = size )
        y = y + np.random.normal(0, noise, size = size )
    X_manifold = np.stack([np.cos(theta), z]).T
    hist, xedges, yedges = np.histogram2d(X_manifold[:, 0], X_manifold[:, 1], bins=(n_bins, n_bins))
    labels = np.arange(0, (n_bins + 1) * (n_bins + 1) ).reshape((n_bins + 1, n_bins + 1))

    cluster_labels = [assign_label(X_manifold[i,:], xedges, yedges, labels) for i in range(X_manifold.shape[0])]
    return np.stack([x,y,z]).T, X_manifold, np.array(cluster_labels)


def create_swissroll(size = 1000, a = 1, b=1, noise= 0,
                  n_bins = 10):
    '''
    create swissroll
    '''
    t = 1.5 * np.pi * (1 + 2 *  np.random.beta(a=a, b=b, size = size))
    x = t * np.cos(t)
    y = 21 *  np.random.beta(a=a, b=b, size = size)
    z = t * np.sin(t)
    # Add noise
    if noise > 0 :
        x = x + np.random.normal(0, noise, size = size )
        y = y + np.random.normal(0, noise, size = size )
        z = z + np.random.normal(0, noise, size = size )
    X_manifold = np.stack([t, y]).T
    hist, xedges, yedges = np.histogram2d(X_manifold[:, 0], X_manifold[:, 1], bins=(n_bins, n_bins))
    labels = np.arange(0, (n_bins + 1) * (n_bins + 1) ).reshape((n_bins + 1, n_bins + 1))
    cluster_labels = [assign_label(X_manifold[i,:], xedges, yedges, labels) for i in range(X_manifold.shape[0])]
    return np.stack([x,y,z]).T, X_manifold, np.array(cluster_labels)

def create_swissroll(size = 1000, a = 1, b=1, noise= 0,
                  n_bins = 10):
    '''
    create swissroll
    '''
    t = 1.5 * np.pi * (1 + 2 *  np.random.beta(a=a, b=b, size = size))
    x = t * np.cos(t)
    y = 21 *  np.random.beta(a=a, b=b, size = size)
    z = t * np.sin(t)
    # Add noise
    if noise > 0 :
        x = x + np.random.normal(0, noise, size = size )
        y = y + np.random.normal(0, noise, size = size )
        z = z + np.random.normal(0, noise, size = size )
    X_manifold = np.stack([t, y]).T
    hist, xedges, yedges = np.histogram2d(X_manifold[:, 0], X_manifold[:, 1], bins=(n_bins, n_bins))
    labels = np.arange(0, (n_bins + 1) * (n_bins + 1) ).reshape((n_bins + 1, n_bins + 1))
    cluster_labels = [assign_label(X_manifold[i,:], xedges, yedges, labels) for i in range(X_manifold.shape[0])]
    return np.stack([x,y,z]).T, X_manifold, np.array(cluster_labels)


def create_trefoil(size = 1000, a = 1, b=1, noise= 0,
                  n_bins = 10):
    '''
    '''
    t =  2 * np.pi * np.random.beta(a=a, b=b, size = size)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = - np.sin(3 * t)
    X_manifold = np.cos(t)
    # Add noise
    if noise > 0 :
        x = x + np.random.normal(0, noise, size = size )
        y = y + np.random.normal(0, noise, size = size )
        z = z + np.random.normal(0, noise, size = size )
    #X_manifold = np.stack([t, y]).T
    # Get the bin edges
    _, bin_edges = np.histogram(X_manifold, bins=n_bins)

    # Assign cluster labels based on bins
    cluster_labels = np.digitize(X_manifold, bin_edges) - 1

    # Ensure the last bin edge captures the upper bound
    cluster_labels[t == bin_edges[-1]] = n_bins - 1
    return np.stack([x,y,z]).T, X_manifold, np.array(cluster_labels)


def create_helix(size = 1000, a = 1, b=1, noise= 0,
                 n_bins = 10, radius_torus=1,
                 radius_tube=0.3, nb_loops=10):
    '''
    create swissroll
    '''
    t =  2 * np.pi * np.random.beta(a=a, b=b, size = size)
    x = (radius_torus + radius_tube * np.cos(nb_loops * t)) * np.cos(t)
    y = (radius_torus + radius_tube * np.cos(nb_loops * t)) * np.sin(t)
    z = radius_tube * np.sin(nb_loops * t)
    # Add noise
    if noise > 0 :
        x = x + np.random.normal(0, noise, size = size )
        y = y + np.random.normal(0, noise, size = size )
        z = z + np.random.normal(0, noise, size = size )
    X_manifold = np.cos(t)
    # Get the bin edges
    _, bin_edges = np.histogram(X_manifold, bins=n_bins)
    # Assign cluster labels based on bins
    cluster_labels = np.digitize(X_manifold, bin_edges) - 1
    # Ensure the last bin edge captures the upper bound
    cluster_labels[X_manifold == bin_edges[-1]] = n_bins - 1
    return np.stack([x,y,z]).T, X_manifold, np.array(cluster_labels)
