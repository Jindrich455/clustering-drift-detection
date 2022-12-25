"""
Implementation of the UCDD algorithm by Dan Shang, Guangquan Zhang and Jie Lu
"""
# TODO: MAKE THIS FILE AS GOOD AS MSSW

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from ucdd import ucdd_supported_parameters as spms


def split_back_to_windows(df_union, labels, len_ref_window, len_test_window):
    """Separate predicted points back to original reference and testing windows (through boolean masks)"""
    ref_mask = pd.Series(np.concatenate([np.repeat(True, len_ref_window), np.repeat(False, len_test_window)]))
    plus_mask = pd.Series(labels, dtype=bool)

    df_ref_plus = df_union[ref_mask & plus_mask]
    df_ref_minus = df_union[ref_mask & (~plus_mask)]
    df_test_plus = df_union[(~ref_mask) & plus_mask]
    df_test_minus = df_union[(~ref_mask) & (~plus_mask)]

    return df_ref_plus, df_ref_minus, df_test_plus, df_test_minus


def join_predict_split(df_ref_window, df_test_window, random_state, metric_id):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    # join the points from two windows
    df_union = pd.concat([df_ref_window, df_test_window])
    df_union_new_index = df_union.set_index(np.arange(len(df_union)))

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, random_state=random_state).fit_predict(df_union_new_index)

    # split values by predicted label and window
    return split_back_to_windows(df_union_new_index, predicted_labels, len(df_ref_window), len(df_test_window))


def compute_neighbors(u, v, metric_id, debug_string='v'):
    neigh = NearestNeighbors(n_neighbors=1, metric=nn_metric_from_id(metric_id), n_jobs=-1)
    neigh.fit(v)

    neigh_ind_v = neigh.kneighbors(u, return_distance=False)
    # print('neigh_ind_' + debug_string, neigh_ind_v)
    unique_v_neighbor_indices = np.unique(neigh_ind_v)
    w = v.iloc[unique_v_neighbor_indices]
    return w


def compute_beta(df_u, df_v0, df_v1, show_2d_plots, debug, metric_id, beta_x=0.5):
    w0 = compute_neighbors(df_u, df_v0, metric_id, 'v0')
    w1 = compute_neighbors(df_u, df_v1, metric_id, 'v1')
    if debug: print('neighbors in W0', len(w0))
    if debug: print('neighbors in W1', len(w1))
    if show_2d_plots:
        ucdd_plotter.plot_u_w0_w1(df_u, w0, w1)
    beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
    beta_additional = scipy.stats.beta.cdf(beta_x, len(w1), len(w0))
    if debug: print('beta', beta)
    return beta, beta_additional


def detect_cd_one_batch(
        df_X_ref,
        df_X_test,
        random_state,
        additional_check,
        metric_id,
        show_2d_plots,
        debug,
        threshold=0.05
):
    """
    Detect whether a concept drift occurred based on one reference and one testing window
    :param df_X_ref:
    :param df_X_test:
    :param random_state:
    :param additional_check:
    :param metric_id:
    :param show_2d_plots:
    :param debug:
    :param threshold:
    :return:
    """
    pass


def all_drifting_batches(
        reference_data_batches,
        testing_data_batches,
        n_clusters=2,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None
):
    """
    Find all drift locations based on the given reference and testing batches

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param n_clusters: desired number of clusters for kmeans
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param coeff: coeff used to detect drift, default=2.66
    :return: a boolean list, length=len(testing_data_batches),
        an entry is True if drift was detected there and False otherwise
    """
    pass

