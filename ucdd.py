"""
Implementation of the UCDD algorithm by Dan Shang, Guangquan Zhang and Jie Lu
"""
import numpy as np
import pandas as pd
import scipy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def join_predict_split(df_ref_window, df_test_window):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    df_ref_window.insert(df_ref_window.shape[1], 'window', 'ref')
    df_test_window.insert(df_test_window.shape[1], 'window', 'test')
    print(df_ref_window)
    df_union = pd.concat([df_ref_window, df_test_window])
    print(df_union)
    df_class_window = df_union[['class', 'window']]
    df_no_labels = df_union.drop(columns=['class', 'window'])
    print(df_no_labels.head())
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df_no_labels)
    labels = KMeans(n_clusters=2, random_state=0).fit_predict(df_no_labels)
    print(kmeans.cluster_centers_)
    print('labels', labels)
    df_predicted_labels = df_no_labels
    df_predicted_labels['labelprediction'] = labels
    df_predicted_labels = df_predicted_labels.join(df_class_window)
    print('df_predicted_labels\n', df_predicted_labels)

    ref_mask = df_predicted_labels['window'] == 'ref'
    plus_mask = df_predicted_labels['labelprediction'] == 1

    ref_plus_mask = ref_mask & plus_mask
    ref_minus_mask = ref_mask & (~plus_mask)
    test_plus_mask = (~ref_mask) & plus_mask
    test_minus_mask = (~ref_mask) & (~plus_mask)

    df_ref_plus = df_predicted_labels[ref_plus_mask]
    df_ref_minus = df_predicted_labels[ref_minus_mask]
    df_test_plus = df_predicted_labels[test_plus_mask]
    df_test_minus = df_predicted_labels[test_minus_mask]

    print('df_ref_plus', df_ref_plus)
    print('df_test_minus', df_test_minus)

    return df_ref_plus, df_ref_minus, df_test_plus, df_test_minus


def get_u_v0_v1(df_u, df_v0, df_v1):
    columns_to_drop = ['labelprediction', 'class', 'window']
    u = df_u.drop(columns=columns_to_drop)
    v0 = df_v0.drop(columns=columns_to_drop)
    v1 = df_v1.drop(columns=columns_to_drop)

    return u, v0, v1


def compute_neighbors(neigh, v, debug_string='v'):
    neigh_dist_v, neigh_ind_v = neigh.kneighbors(v)
    print('neigh_ind_' + debug_string, neigh_ind_v)
    unique_v_neighbor_indices = np.unique(neigh_ind_v)
    w = v.iloc[unique_v_neighbor_indices]
    return w


def compute_beta(df_u, df_v0, df_v1, beta_x=0.5):
    u, v0, v1 = get_u_v0_v1(df_u, df_v0, df_v1)
    print('v0\n', v0)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(u)
    w0 = compute_neighbors(neigh, v0, 'v0')
    w1 = compute_neighbors(neigh, v1, 'v1')
    print('w0', w0)
    print('w1', w1)
    beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
    print('beta', beta)
    return beta


def detect_cd(df_ref_window, df_test_window, threshold=0.05):
    """Detect whether a concept drift occurred based on a reference and a testing window"""
    df_ref_plus, df_ref_minus, df_test_plus, df_test_minus = join_predict_split(df_ref_window, df_test_window)
    beta_minus = compute_beta(df_ref_plus, df_ref_minus, df_test_minus)
    beta_plus = compute_beta(df_ref_minus, df_ref_plus, df_test_plus)
    if beta_plus < threshold or beta_minus < threshold:
        return True
    else:
        return False


def drift_occurrences_list(reference_batch_list, test_batch_list):
    """Return a list of all batches where the algorithm detected drift"""
