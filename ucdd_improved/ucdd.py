"""
Implementation of the UCDD algorithm by Dan Shang, Guangquan Zhang and Jie Lu
"""
import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import scipy
from pyclustering.utils import distance_metric, type_metric
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from ucdd_improved import ucdd_supported_parameters as spms


def split_back_to_windows(window_union, labels, len_ref_window, len_test_window):
    """Separate predicted points back to original reference and testing windows (through boolean masks)"""
    ref_mask = np.concatenate([np.repeat(True, len_ref_window), np.repeat(False, len_test_window)])
    plus_mask = np.where(labels == 1, True, False)

    ref_plus = window_union[np.logical_and(ref_mask, plus_mask)]
    ref_minus = window_union[np.logical_and(ref_mask, np.logical_not(plus_mask))]
    test_plus = window_union[np.logical_and(np.logical_not(ref_mask), plus_mask)]
    test_minus = window_union[np.logical_and(np.logical_not(ref_mask), np.logical_not(plus_mask))]

    return ref_plus, ref_minus, test_plus, test_minus


def join_predict_split(ref_window, test_window, random_state,
                       n_init, max_iter, tol):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    # join the points from two windows
    window_union = np.vstack((ref_window, test_window))

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)\
        .fit_predict(window_union)

    # split values by predicted label and window
    return split_back_to_windows(window_union, predicted_labels, ref_window.shape[0], test_window.shape[0])


def compute_neighbors(u, v, debug_string='v'):
    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neigh.fit(v)

    neigh_ind_v = neigh.kneighbors(u, return_distance=False)
    # print('neigh_ind_' + debug_string, neigh_ind_v)
    unique_v_neighbor_indices = np.unique(neigh_ind_v)
    # print('unique neigh_ind_' + debug_string, unique_v_neighbor_indices)
    w = v[unique_v_neighbor_indices]
    return w


def compute_beta(u, v0, v1, beta_x=0.5, debug=False):
    w0 = compute_neighbors(u, v0, 'v0')
    w1 = compute_neighbors(u, v1, 'v1')
    if debug: print('neighbors in W0', len(w0))
    if debug: print('neighbors in W1', len(w1))
    beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
    beta_additional = scipy.stats.beta.cdf(beta_x, len(w1), len(w0))
    if debug: print('beta', beta)
    return beta, beta_additional


def concept_drift_detected(
        ref_window,
        test_window,
        random_state,
        additional_check,
        n_init,
        max_iter,
        tol,
        threshold=0.05,
        debug=False,
):
    """Detect whether a concept drift occurred based on a reference and a testing window"""
    ref_plus, ref_minus, test_plus, test_minus = \
        join_predict_split(ref_window, test_window, random_state,
                           n_init=n_init, max_iter=max_iter, tol=tol)

    if debug: print('BETA MINUS (ref+, ref-, test-)')
    beta_minus, beta_minus_additional = compute_beta(
        ref_plus, ref_minus, test_minus)
    if debug: print('BETA PLUS (ref-, ref+, test+)')
    beta_plus, beta_plus_additional = compute_beta(
        ref_minus, ref_plus, test_plus)

    drift = (beta_plus < threshold or beta_minus < threshold)
    if additional_check:
        drift = drift | (beta_plus_additional < threshold or beta_minus_additional < threshold)

    return drift


def all_drifting_batches(
        reference_data_batches,
        testing_data_batches,
        train_batch_strategy,
        additional_check,
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
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :return: a boolean list, length=len(testing_data_batches),
        an entry is True if drift was detected there and False otherwise
    """

    drifts_detected = []
    for i, test_window in enumerate(testing_data_batches):
        print('#### TEST BATCH', i, 'of', len(testing_data_batches), '####')
        num_ref_drifts = 0 # how many training batches signal drift against this testing batch
        for j, ref_window in enumerate(reference_data_batches):
            drift_here = concept_drift_detected(
                ref_window, test_window, random_state, additional_check, n_init, max_iter, tol)
            if drift_here:
                num_ref_drifts += 1
        if train_batch_strategy == spms.TrainBatchStrategies.ANY:
            drift = num_ref_drifts > 0
        elif train_batch_strategy == spms.TrainBatchStrategies.MAJORITY:
            drift = num_ref_drifts > (len(reference_data_batches) / 2)
        elif train_batch_strategy == spms.TrainBatchStrategies.ALL:
            drift = num_ref_drifts == len(reference_data_batches)
        else:
            raise NameError('The train batch strategy', train_batch_strategy, 'is not supported')
        drifts_detected.append(drift)

    return drifts_detected

