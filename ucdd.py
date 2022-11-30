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
    # print('len_ref_window', len_ref_window)
    # print('len_test_window', len_test_window)
    # print('plus_mask', plus_mask)

    # print('\n\n\n')
    # print('\nref_mask len=', len(ref_mask), ', head:\n', ref_mask.head())
    # print('\nplus_mask len=', len(plus_mask), ', head:\n', plus_mask.head())
    # print('\ndf_union len=', len(df_union), ', head:\n', df_union.head())
    #
    # ref_plus_mask = ref_mask & plus_mask
    # print('\nref_plus_mask len=', len(ref_plus_mask), ', head:\n', ref_plus_mask.head())
    #
    # print('\nref_mask index', ref_mask.index)
    # print('\ndf_union index', df_union.index)

    # df_ref = df_union.loc[ref_mask]
    # print('\ndf_ref len=', len(df_ref), ', head:\n', df_ref.head())

    # df_plus = df_union.loc[plus_mask]
    # print('\ndf_plus len=', len(df_plus), ', head:\n', df_plus.head())
    #
    # df_ref_plus2 = df_union.loc[ref_plus_mask]
    # print('\ndf_ref_plus2 len=', len(df_ref_plus2), ', head:\n', df_ref_plus2.head())

    df_ref_plus = df_union[ref_mask & plus_mask]
    df_ref_minus = df_union[ref_mask & (~plus_mask)]
    df_test_plus = df_union[(~ref_mask) & plus_mask]
    df_test_minus = df_union[(~ref_mask) & (~plus_mask)]

    print('df_ref_plus len', len(df_ref_plus))
    print('df_ref_minus len', len(df_ref_minus))
    print('df_test_plus len', len(df_test_plus))
    print('df_test_minus len', len(df_test_minus))

    return df_ref_plus, df_ref_minus, df_test_plus, df_test_minus


def join_predict_split(df_ref_window, df_test_window):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    # print('df_ref_window index', df_ref_window.index)
    # print('df_test_window index', df_test_window.index)

    # join the points from two windows
    df_union = pd.concat([df_ref_window, df_test_window])
    # print('df_union index', df_union.index)
    df_union_new_index = df_union.set_index(np.arange(len(df_union)))
    # print('df_union NEW index', df_union_new_index.index)

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, random_state=0).fit_predict(df_union_new_index)
    # print('predicted labels', predicted_labels)

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


def compute_beta(df_u, df_v0, df_v1, beta_x=0.5):
    # print('df_u index', df_u.index)
    # print('dv_v0 index', df_v0.index)
    # print('dv_v1 index', df_v1.index)
    w0 = compute_neighbors(df_u, df_v0, 'v0')
    w1 = compute_neighbors(df_u, df_v1, 'v1')
    print('neighbors in W0', len(w0))
    print('neighbors in W1', len(w1))
    beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
    print('beta', beta)
    return beta


def detect_cd(df_X_ref, df_X_test, threshold=0.05):
    """Detect whether a concept drift occurred based on a reference and a testing window"""
    df_ref_plus, df_ref_minus, df_test_plus, df_test_minus = \
        join_predict_split(df_X_ref, df_X_test)
    print('BETA MINUS (ref+, ref-, test-)')
    beta_minus = compute_beta(df_ref_plus, df_ref_minus, df_test_minus)
    print('BETA PLUS (ref-, ref+, test+)')
    beta_plus = compute_beta(df_ref_minus, df_ref_plus, df_test_plus)
    if beta_plus < threshold or beta_minus < threshold:
        return True
    else:
        return False


def drift_occurrences_list(X_ref_batches, X_test_batches):
    """Return a list of all batches where the algorithm detected drift"""
    # I am trying to use only one testing batch
    df_X_ref = X_ref_batches[-1]
    drift_signal_locations = []
    for i, df_X_test in enumerate(X_test_batches):
        print('#### BATCH', i, '####')
        if detect_cd(df_X_ref, df_X_test):
            drift_signal_locations.append(i)
        print('\n\n')
    return drift_signal_locations
