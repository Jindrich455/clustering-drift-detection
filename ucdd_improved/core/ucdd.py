"""
Drift detection algorithm from
[1] D. Shang, G. Zhang, and J. Lu,
“Fast concept drift detection using unlabeled data,”
in Developments of Artificial Intelligence Technologies in Computation and Robotics,
Cologne, Germany, Oct. 2020, pp. 133–140. doi: 10.1142/9789811223334_0017.

- Unless specified otherwise, functions in this file work with numpy arrays
- The terms "batch" and "window" mean the same thing
"""
import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import scipy
from pyclustering.utils import distance_metric, type_metric
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from . import ucdd_supported_parameters as spms


def split_back_to_windows(window_union, labels, len_ref_window, len_test_window):
    """
    Separate predicted points back to original reference and testing windows (through boolean masks)

    :param window_union: array of shape (len_ref_window + len_test_window, #attributes)
    :param labels: list of cluster predictions of points in window_union
    :param len_ref_window: #points in the reference window
    :param len_test_window: #points in the testing window
    :return: 2d arrays of X0+, X0-, X1+, X1-
    """
    ref_mask = np.concatenate([np.repeat(True, len_ref_window), np.repeat(False, len_test_window)])
    plus_mask = np.where(labels == 1, True, False)

    ref_plus = window_union[np.logical_and(ref_mask, plus_mask)]
    ref_minus = window_union[np.logical_and(ref_mask, np.logical_not(plus_mask))]
    test_plus = window_union[np.logical_and(np.logical_not(ref_mask), plus_mask)]
    test_minus = window_union[np.logical_and(np.logical_not(ref_mask), np.logical_not(plus_mask))]

    return ref_plus, ref_minus, test_plus, test_minus


def join_predict_split(ref_window, test_window,
                       n_init, max_iter, tol, random_state):
    """
    Join points from two windows, predict their labels through kmeans, then separate them again

    :param ref_window: array of shape (#points in ref_window, #attributes)
    :param test_window: array of shape (#points in test_window, #attributes)
    :param n_init: see sklearn.cluster.KMeans n_init
    :param max_iter: see sklearn.cluster.KMeans max_iter
    :param tol: see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :return: 2d arrays of X0+, X0-, X1+, X1-
    """
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    # join the points from two windows
    window_union = np.vstack((ref_window, test_window))

    print('n_init', n_init, 'max_iter', max_iter, 'tol', tol)

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)\
        .fit_predict(window_union)

    # split values by predicted label and window
    return split_back_to_windows(window_union, predicted_labels, ref_window.shape[0], test_window.shape[0])


def compute_neighbors(u, v, debug_string='v'):
    """
    Find the indices of nearest neighbors of v in u

    :param u: array of shape (#points in u, #attributes),
    :param v: array of shape (#points in v, #attributes),
    :param debug_string: string to use in debug print statements
    :return: array of shape (#unique nearest neighbour indices of v in u, 1)
    """
    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neigh.fit(v)

    neigh_ind_v = neigh.kneighbors(u, return_distance=False)
    # print('neigh_ind_' + debug_string, neigh_ind_v)
    unique_v_neighbor_indices = np.unique(neigh_ind_v)
    # print('unique neigh_ind_' + debug_string, unique_v_neighbor_indices)
    w = v[unique_v_neighbor_indices]
    return w


def compute_beta(u, v0, v1, beta_x=0.5, debug=False):
    """
    Find neighbors and compute beta based on u, v0, v1

    :param u: array of shape (#points in u, #attributes), cluster U from the algorithm
    :param v0: array of shape (#points in v0, #attributes), cluster V0 from the algorithm
    :param v1: array of shape (#points in v1, #attributes), cluster V1 from the algorithm
    :param beta_x: default=0.5, x to use in the Beta distribution
    :param debug: debug: flag for helpful print statements
    :return: (beta - the regular beta cdf value,
        beta_additional - the beta cdf value for exchanged numbers of neighbours as parameters)
    """
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
        additional_check,
        n_init,
        max_iter,
        tol,
        random_state,
        threshold=0.05,
        debug=False,
):
    """
    Detect whether a concept drift occurred based on one reference and one testing window

    :param ref_window: array of shape (#points in this reference window, #attributes)
    :param test_window: array of shape (#points in this testing window, #attributes)
    :param additional_check: whether to use a two-fold test or not
    :param n_init: see sklearn.cluster.KMeans n_init
    :param max_iter: see sklearn.cluster.KMeans max_iter
    :param tol: see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param threshold: default=0.05, statistical threshold to detect drift
    :param debug: flag for helpful print statements
    :return: true if drift is detected based on the two windows, false otherwise
    """
    print('BEFORE SPLITTING TO CLUSTERS')
    ref_plus, ref_minus, test_plus, test_minus = \
        join_predict_split(ref_window, test_window,
                           n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)

    print('CLUSTERS - numpy shapes:')
    print('ref_plus', ref_plus.shape)
    print('ref_minus', ref_minus.shape)
    print('test_plus', test_plus.shape)
    print('test_minus', test_minus.shape)

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
    :param train_batch_strategy: the reference batch strategy for drift detection
    :param additional_check: whether to use a two-fold test or not
    :param n_init: default=10, see sklearn.cluster.KMeans n_init
    :param max_iter: default=300, see sklearn.cluster.KMeans max_iter
    :param tol: default=1e-4, see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :return: a boolean list, length=len(testing_data_batches),
        an entry is True if drift was detected there and False otherwise
    """

    print('random_state')
    print(random_state)
    drifts_detected = []
    for i, test_window in enumerate(testing_data_batches):
        # print('#### TEST BATCH', i, 'of', len(testing_data_batches), '####')
        num_ref_drifts = 0 # how many training batches signal drift against this testing batch
        for j, ref_window in enumerate(reference_data_batches):
            drift_here = concept_drift_detected(
                ref_window, test_window, additional_check, n_init, max_iter, tol, random_state)
            if drift_here:
                num_ref_drifts += 1
        if train_batch_strategy == spms.TrainBatchStrategies.ANY:
            drift = num_ref_drifts > 0
        elif train_batch_strategy == spms.TrainBatchStrategies.SUBMAJORITY:
            drift = num_ref_drifts >= (len(reference_data_batches) // 2)
        elif train_batch_strategy == spms.TrainBatchStrategies.MAJORITY:
            drift = num_ref_drifts > (len(reference_data_batches) / 2)
        elif train_batch_strategy == spms.TrainBatchStrategies.ALL:
            drift = num_ref_drifts == len(reference_data_batches)
        else:
            raise NameError('The train batch strategy', train_batch_strategy, 'is not supported')
        drifts_detected.append(drift)

    return drifts_detected

