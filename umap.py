import numpy as np


def prob_high_dim(sigma, dist):
    """
    For each row of Euclidean distance matrix (dist_row) compute
    probability in high dimensions (1D array)
    """
    d = dist - np.min(dist); d[d < 0] = 0
    return np.exp(- d / sigma)

def k(prob):
    """
    Compute n_neighbor = k (scalar) for each 1D array of high-dimensional probability
    """
    return np.power(2, np.sum(prob))


def sigma_binary_search(k_of_sigma, fixed_k):
    """
    Solve equation k_of_sigma(sigma) = fixed_k
    with respect to sigma by the binary search algorithm
    Do we really need this?
    """
    sigma_lower_limit = 0;
    sigma_upper_limit = 100;
    for i in range(20):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        if k_of_sigma(approx_sigma) < fixed_k:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
            break
    return approx_sigma


def prob_low_dim(Y, YY, a=1., b=1.):
    """
    Compute matrix of probabilities q_ij in low-dimensional space
    """
    inv_distances = torch.power(1 + a * torch.sum(torch.square(Y-YY))**b, -1)
    return inv_distances
