import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import time as tm
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr


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
    for i in np.concatenate((np.arange(0, (t - t_0) - (v_length - 1)),
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


def single_CCM(df, x_ID, y_ID,
               L_step=5, E=3, taxonomy="genus",
               print_timeit=False, print_results=False, plot_result=False):
    '''
    '''
    how_long_metadata = np.count_nonzero(np.isnan(df.index.values))
    first_day = df.index.values[how_long_metadata]
    subject, sample_loc = df.loc[first_day, "common_sample_site"], df.loc[first_day, "host_individual"]
    x_name, y_name = df.loc[df['sample_name'] == taxonomy].loc[:, [x_ID, y_ID]].values[0]
    x_data, y_data = df.loc[first_day:, x_ID].values, df.loc[first_day:, y_ID].values

    # Compute the time series length
    L_max = len(x_data)
    L_step = L_step
    L = np.arange(E * 2 + 1, L_max, L_step)
    if (L[-1] != len(x_data)):
        L = np.append(L, len(x_data))

    return CCM_result(x_data, y_data, x_ID, y_ID, x_name, y_name, L,
                      E=E, subject=subject, sample_loc=sample_loc,
                      print_timeit=print_timeit, print_results=print_results, plot_result=plot_result)


def CCM_result(x_data, y_data, x_ID, y_ID, x_name, y_name, L,
               E=3, subject="M3", sample_loc="feces",
               print_timeit=False, print_results=False, plot_result=False):
    '''
    Args:
        x_data (np.array): the raw data array used to approximate y
        y_data (np.array): the raw data array which we plan to approximate using CCM
    Returns:
    '''
    start_time = tm.time()  # in case we want to time the function

    corr_X_xmap_Y, corr_Y_xmap_X = [], []
    L_step = L[1] - L[0]

    for i in L:
        x_orig = x_data[:i]
        y_orig = y_data[:i]

        Y_approx = generateYApprox(x_orig, y_orig, E=E, how_long=0)
        corr_Y_xmap_X.append(pearsonr(y_orig[(E - 1):i], Y_approx)[0])

        X_approx = generateYApprox(y_orig, x_orig, E=E, how_long=0)
        corr_X_xmap_Y.append(pearsonr(x_orig[(E - 1):], X_approx)[0])

    spearmanXY, spearmanYX = spearmanr(corr_X_xmap_Y, L), spearmanr(corr_Y_xmap_X, L)

    if (print_timeit):
        end_time = tm.time()
        print("Loop for IDs (%s, %s) took %.3f seconds." % (x_ID, y_ID, end_time - start_time))

    if (print_results):
        print("%s xmap %s: spearman_coeff = %.4f" % (x_name, y_name, spearmanXY[0]))
        print("%s xmap %s: spearman_coeff = %.4f" % (y_name, x_name, spearmanYX[0]))

    if (plot_result):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(L, corr_Y_xmap_X, 'g')
        plt.plot(L, corr_X_xmap_Y, 'b')
        plt.title("CCM for %s and %s (columns %s and %s, respect.)" % (x_name, y_name, x_ID, y_ID))
        plt.legend(["%s xmap %s" % (x_name, y_name), "%s xmap %s" % (y_name, x_name)])
        plt.xlabel("L (length of time series considered)")
        plt.ylabel("rho (Pearson correlation coeff.)")
        plt.show()


    df_result = pd.DataFrame({"x_ID": [x_ID, y_ID], "y_ID": [y_ID, x_ID],
                              "x_name": [x_name, y_name], "y_name": [y_name, x_name],
                              "spearman_coeff": [spearmanXY[0], spearmanYX[0]],
                              "spearman_coeff_p": [spearmanXY[1], spearmanYX[1]],
                              "pearson_coeff": [corr_X_xmap_Y[-1], corr_Y_xmap_X[-1]],
                              "L": [L[-1], L[-1]],
                              "subject": [subject, subject], "sample_loc": [sample_loc, sample_loc],
                              "L_step": [L_step, L_step]})

    return df_result
