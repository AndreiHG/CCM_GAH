import numpy as np
import scipy.spatial.distance as dist
from scipy.stats.stats import pearsonr


def shadowManifold(X, E, tau=1):
    '''
    Creates a manifold of the time series X = {X(1), X(2), ..., X(L)}
    using time steps tau and E dimensionality.
    '''
    t_0 = 0 + (E - 1) * tau
    lagged_times = np.arange(t_0, len(X), tau)
    shadow_M = []
    for i in lagged_times:
        shadow_M.append([X[np.arange(i, i - (E - 1) * tau - 1, -tau)], i])  # goes from i to 0

    return shadow_M

def nearestNeighbours(shadow_M, t, n_neigh):
    '''
    Returns the positions (index) of n-neigh nearest neighbours of x(t) the shadow
    manifold shadow_M, in order from closest to farthest.
    '''
    dist_list = []
    t_0 = shadow_M[0][1]
    if (t - t_0) < 0:
        print("The vector x(t) for your particular choice of time t doesn't exist in the shadow manifold M_X")
        return None
    v = shadow_M[t - shadow_M[0][1]][0]
    distances = np.zeros(len(shadow_M))
    for i in range(len(shadow_M)):
        distances[i] = dist.euclidean(shadow_M[i][0], v)
    # distances = np.array(dist_list)
    # print(dist_list)
    # print(np.argsort(distances)[1:n_neigh+1]+t_0, np.sort(distances)[1:n_neigh+1])

    return [shadow_M[i] for i in np.argsort(distances)[1:n_neigh + 1]]

def nearestLeaveOutNeighbours(shadow_M, t, n_neigh):
    '''
    Returns the positions (index) of n-neigh nearest neighbours of x(t) the shadow
    manifold shadow_M, in order from closest to farthest.
    '''
    dist_list = []
    t_0 = shadow_M[0][1]
    if (t - t_0) < 0:
        print("The vector x(t) for your particular choice of time t doesn't exist in the shadow manifold M_X")
        return None
    v = shadow_M[t - t_0][0]
    v_length = len(v)
    distances = np.zeros(len(shadow_M)) + 9999
    for i in np.concatenate((np.arange(0, (t - t_0) - (v_length - 1)), \
                             np.arange((t - t_0) + (v_length - 1) + 1, len(shadow_M)))):
        distances[i] = dist.euclidean(shadow_M[i][0], v)

    return [shadow_M[i] for i in np.argsort(distances)[0:n_neigh]]

def weights(v_t, neigh):
    distances_neigh, u = np.zeros(len(neigh)), np.zeros(len(neigh))
    for i in range(len(neigh)):
        distances_neigh[i] = dist.euclidean(v_t[0], neigh[i][0])
        u[i] = np.exp(-distances_neigh[i])  # /distances_neigh[0])
    return u / np.sum(u)

def generateYApprox(X, Y, E, how_long, tau=1, leaveOut=False):
    '''
    Generates the time series Y_tilde (or Y_approximative), cross mapped using time series X,
    truncated to how_long many elements.
    '''

    shadow_M_X = shadowManifold(X, E, tau)
    neigh, weigh = [], []

    Y_tilde = np.zeros(len(shadow_M_X))
    Y_neigh_index = np.zeros(E + 1, dtype=int)
    t_0 = shadow_M_X[0][1]  # I think...
    for i in range(len(shadow_M_X)):
        if (leaveOut):
            neigh = nearestLeaveOutNeighbours(shadow_M_X, shadow_M_X[i][1], E + 1)
        else:
            neigh = nearestNeighbours(shadow_M_X, shadow_M_X[i][1], E + 1)
        weigh = weights(shadow_M_X[i], neigh)
        for j in range(len(neigh)):
            Y_neigh_index[j] = neigh[j][1]  # - t_0

        Y_tilde[i] = np.sum(np.multiply(weigh, Y[Y_neigh_index]))

    return Y_tilde
