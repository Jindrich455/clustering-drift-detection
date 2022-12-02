"""
Implementation of the UCDD algorithm by Dan Shang, Guangquan Zhang and Jie Lu
"""
import numpy as np
import pandas as pd
import scipy
import supported_parameters as spms
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import ucdd_plotter


def split_back_to_windows(df_union, labels, len_ref_window, len_test_window):
    """Separate predicted points back to original reference and testing windows (through boolean masks)"""
    ref_mask = pd.Series(np.concatenate([np.repeat(True, len_ref_window), np.repeat(False, len_test_window)]))
    plus_mask = pd.Series(labels, dtype=bool)

    df_ref_plus = df_union[ref_mask & plus_mask]
    df_ref_minus = df_union[ref_mask & (~plus_mask)]
    df_test_plus = df_union[(~ref_mask) & plus_mask]
    df_test_minus = df_union[(~ref_mask) & (~plus_mask)]

    return df_ref_plus, df_ref_minus, df_test_plus, df_test_minus


def join_predict_split(df_ref_window, df_test_window, random_state):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    # join the points from two windows
    df_union = pd.concat([df_ref_window, df_test_window])
    df_union_new_index = df_union.set_index(np.arange(len(df_union)))

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, random_state=random_state).fit_predict(df_union_new_index)

    # split values by predicted label and window
    return split_back_to_windows(df_union_new_index, predicted_labels, len(df_ref_window), len(df_test_window))


def compute_neighbors(u, v, debug_string='v'):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(v)

    neigh_ind_v = neigh.kneighbors(u, return_distance=False)
    # print('neigh_ind_' + debug_string, neigh_ind_v)
    unique_v_neighbor_indices = np.unique(neigh_ind_v)
    w = v.iloc[unique_v_neighbor_indices]
    return w


def compute_beta(df_u, df_v0, df_v1, show_2d_plots, debug, beta_x=0.5):
    w0 = compute_neighbors(df_u, df_v0, 'v0')
    w1 = compute_neighbors(df_u, df_v1, 'v1')
    if debug: print('neighbors in W0', len(w0))
    if debug: print('neighbors in W1', len(w1))
    if show_2d_plots:
        ucdd_plotter.plot_u_w0_w1(df_u, w0, w1)
    beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
    if debug: print('beta', beta)
    return beta


def detect_cd(df_X_ref, df_X_test, random_state, show_2d_plots, additional_check, debug, threshold=0.05):
    """Detect whether a concept drift occurred based on a reference and a testing window"""
    df_ref_plus, df_ref_minus, df_test_plus, df_test_minus = \
        join_predict_split(df_X_ref, df_X_test, random_state)

    if show_2d_plots:
        ucdd_plotter.plot_predicted(df_ref_plus, df_ref_minus, df_test_plus, df_test_minus)

    if debug: print('BETA MINUS (ref+, ref-, test-)')
    beta_minus = compute_beta(df_ref_plus, df_ref_minus, df_test_minus, show_2d_plots, debug)
    if debug: print('BETA PLUS (ref-, ref+, test+)')
    beta_plus = compute_beta(df_ref_minus, df_ref_plus, df_test_plus, show_2d_plots, debug)

    drift = False
    if beta_plus < threshold or beta_minus < threshold:
        drift = True
    if additional_check and ((1 - beta_plus) < threshold or (1 - beta_minus) < threshold):
        drift = True
    return drift


def drift_occurrences_list(
        X_ref_batches, X_test_batches, random_state, additional_check, detect_all_training_batches,
        show_2d_plots=False, debug=False):
    """Return a list of all batches where the algorithm detected drift"""
    drift_signal_locations = []
    for i, df_X_test in enumerate(X_test_batches):
        print('#### TEST BATCH', i, 'of', len(X_test_batches), '####')
        drift = False
        if detect_all_training_batches:
            drift = True
        for j, df_X_ref in enumerate(X_ref_batches):
            if debug: print('--- training batch', j, '---')
            drift_here = detect_cd(df_X_ref, df_X_test, random_state, show_2d_plots, additional_check, debug)
            if detect_all_training_batches:
                drift = drift & drift_here
            else:
                drift = drift | drift_here
        if drift:
            drift_signal_locations.append(i)
        if debug: print('\n\n')
    return drift_signal_locations
