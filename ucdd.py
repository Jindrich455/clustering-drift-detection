"""
Implementation of the UCDD algorithm by Dan Shang, Guangquan Zhang and Jie Lu
"""
import numpy as np
import pandas as pd
import scipy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def split_back_to_windows(df_union, labels, len_ref_window, len_test_window):
    """Separate predicted points back to original reference and testing windows (through boolean masks)"""
    ref_mask = pd.Series(np.concatenate([np.repeat(True, len_ref_window), np.repeat(False, len_test_window)]))
    plus_mask = pd.Series(labels, dtype=bool)

    df_ref_plus = df_union[ref_mask & plus_mask]
    df_ref_minus = df_union[ref_mask & (~plus_mask)]
    df_test_plus = df_union[(~ref_mask) & plus_mask]
    df_test_minus = df_union[(~ref_mask) & (~plus_mask)]

    return df_ref_plus, df_ref_minus, df_test_plus, df_test_minus


def join_predict_split(df_ref_window, df_test_window):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    # join the points from two windows
    df_union = pd.concat([df_ref_window, df_test_window])

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, random_state=0).fit_predict(df_union)

    # split values by predicted label and window
    return split_back_to_windows(df_union, predicted_labels, len(df_ref_window), len(df_test_window))


def compute_neighbors(neigh, v, debug_string='v'):
    neigh_dist_v, neigh_ind_v = neigh.kneighbors(v)
    print('neigh_ind_' + debug_string, neigh_ind_v)
    unique_v_neighbor_indices = np.unique(neigh_ind_v)
    w = v.iloc[unique_v_neighbor_indices]
    return w


def compute_beta(df_u, df_v0, df_v1, beta_x=0.5):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(df_u)
    w0 = compute_neighbors(neigh, df_v0, 'v0')
    w1 = compute_neighbors(neigh, df_v1, 'v1')
    beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
    print('beta', beta)
    return beta


def detect_cd(df_X_ref, df_X_test, threshold=0.05):
    """Detect whether a concept drift occurred based on a reference and a testing window"""
    df_ref_plus, df_ref_minus, df_test_plus, df_test_minus = \
        join_predict_split(df_X_ref, df_X_test)
    beta_minus = compute_beta(df_ref_plus, df_ref_minus, df_test_minus)
    beta_plus = compute_beta(df_ref_minus, df_ref_plus, df_test_plus)
    if beta_plus < threshold or beta_minus < threshold:
        return True
    else:
        return False


def drift_occurrences_list(reference_batch_list, test_batch_list):
    """Return a list of all batches where the algorithm detected drift"""