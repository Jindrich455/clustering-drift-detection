"""
Drift detection algorithm from
[1] D. Shang, G. Zhang, and J. Lu,
“Fast concept drift detection using unlabeled data,”
in Developments of Artificial Intelligence Technologies in Computation and Robotics,
Cologne, Germany, Oct. 2020, pp. 133–140. doi: 10.1142/9789811223334_0017.

@author: Jindrich POHL

- Unless specified otherwise, functions in this file work with numpy arrays
- The terms "batch" and "window" mean the same thing
"""
import itertools

import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import scipy
from pyclustering.utils import distance_metric, type_metric
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from . import ucdd_supported_parameters as spms
from multiprocessing import Pool


def print_label_cluster_stats(actual_label_cluster, cluster_name):
    cluster_size = np.shape(actual_label_cluster)[0]
    print('number of points in the', cluster_name, 'cluster:', cluster_size)
    class1_perc = 100 * np.sum(actual_label_cluster) / cluster_size
    print('actual class 1 percentage in the', cluster_name, 'cluster:',
          class1_perc)
    print('actual class 0 percentage in the', cluster_name, 'cluster:',
          100 - class1_perc)


def split_back_to_windows(window_union, labels, len_ref_window, len_test_window, label_batch_union=None):
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

    ref_plus_mask = np.logical_and(ref_mask, plus_mask)
    ref_minus_mask = np.logical_and(ref_mask, np.logical_not(plus_mask))
    test_plus_mask = np.logical_and(np.logical_not(ref_mask), plus_mask)
    test_minus_mask = np.logical_and(np.logical_not(ref_mask), np.logical_not(plus_mask))

    cluster_classif_acc = None

    if label_batch_union is not None:
        total_points = len_ref_window + len_test_window
        print('label batch union shape')
        print(label_batch_union.shape)
        print('plus mask shape')
        print(plus_mask.shape)
        num_class1_in_plus = np.count_nonzero(label_batch_union[plus_mask])
        num_class0_in_minus = np.count_nonzero(label_batch_union[~plus_mask] == 0)
        perc_correctly_classified_points_v1 = (num_class1_in_plus + num_class0_in_minus) / total_points

        num_class0_in_plus = np.count_nonzero(label_batch_union[plus_mask] == 0)
        num_class1_in_minus = np.count_nonzero(label_batch_union[~plus_mask])
        perc_correctly_classified_points_v2 = (num_class0_in_plus + num_class1_in_minus) / total_points

        cluster_classif_acc = max(perc_correctly_classified_points_v1,
                                  perc_correctly_classified_points_v2)

        ref_plus_actual_labels = label_batch_union[ref_plus_mask]
        ref_minus_actual_labels = label_batch_union[ref_minus_mask]
        test_plus_actual_labels = label_batch_union[test_plus_mask]
        test_minus_actual_labels = label_batch_union[test_minus_mask]

        # print_label_cluster_stats(ref_plus_actual_labels, 'ref plus')
        # print_label_cluster_stats(ref_minus_actual_labels, 'ref minus')
        # print_label_cluster_stats(test_plus_actual_labels, 'test plus')
        # print_label_cluster_stats(test_minus_actual_labels, 'test minus')

    ref_plus = window_union[ref_plus_mask]
    ref_minus = window_union[ref_minus_mask]
    test_plus = window_union[test_plus_mask]
    test_minus = window_union[test_minus_mask]

    return ref_plus, ref_minus, test_plus, test_minus, cluster_classif_acc


def join_predict_split(ref_window, test_window,
                       n_init, max_iter, tol, random_state,
                       reference_label_batch=None,
                       testing_label_batch=None):
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

    # print('n_init', n_init, 'max_iter', max_iter, 'tol', tol)

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)\
        .fit_predict(window_union)

    label_batch_union = None
    if reference_label_batch is not None and testing_label_batch is not None:
        label_batch_union = np.vstack((reference_label_batch, testing_label_batch))

    # split values by predicted label and window
    return split_back_to_windows(window_union, predicted_labels, ref_window.shape[0], test_window.shape[0],
                                 label_batch_union)


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
    # if there is so much imbalance that at least one cluster is empty, report drift immediately
    if min(len(u), len(v0), len(v1)) == 0:
        beta = 0
        beta_additional = 0
    else:
        w0 = compute_neighbors(u, v0, 'v0')
        w1 = compute_neighbors(u, v1, 'v1')
        if debug: print('neighbors in W0', len(w0))
        if debug: print('neighbors in W1', len(w1))
        beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
        beta_additional = scipy.stats.beta.cdf(beta_x, len(w1), len(w0))
        if debug: print('beta', beta)
        if debug: print('beta additional', beta_additional)
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
        reference_label_batch=None,
        testing_label_batch=None
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
    ref_plus, ref_minus, test_plus, test_minus, cluster_classif_acc = \
        join_predict_split(ref_window, test_window,
                           n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state,
                           reference_label_batch=reference_label_batch, testing_label_batch=testing_label_batch)

    if debug: print('BETA MINUS (ref+, ref-, test-)')
    beta_minus, beta_minus_additional = compute_beta(
        ref_plus, ref_minus, test_minus, debug=debug)
    if debug: print('BETA PLUS (ref-, ref+, test+)')
    beta_plus, beta_plus_additional = compute_beta(
        ref_minus, ref_plus, test_plus, debug=debug)

    drift = (beta_plus < threshold or beta_minus < threshold)
    if additional_check:
        drift = drift | (beta_plus_additional < threshold or beta_minus_additional < threshold)

    return drift, cluster_classif_acc


def all_drifting_batches(
        reference_data_batches,
        testing_data_batches,
        min_ref_batches_drift,
        additional_check,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        parallel=True,
        reference_label_batches=None,
        testing_label_batches=None,
        debug=False
):
    """
    Find all drift locations based on the given reference and testing batches

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param min_ref_batches_drift: the minimum fraction of reference batches that must signal drift for one test batch
    :param additional_check: whether to use a two-fold test or not
    :param n_init: default=10, see sklearn.cluster.KMeans n_init
    :param max_iter: default=300, see sklearn.cluster.KMeans max_iter
    :param tol: default=1e-4, see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :return: a boolean list, length=len(testing_data_batches),
        an entry is True if drift was detected there and False otherwise
    """

    if parallel:
        drifts_detected = all_drifting_batches_parallel(
            reference_data_batches,
            testing_data_batches,
            min_ref_batches_drift,
            additional_check,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    else:
        print('random_state')
        print(random_state)
        drifts_detected = []
        for i, test_window in enumerate(testing_data_batches):
            print('\n\n#### TEST BATCH', i, 'of', len(testing_data_batches), '####')
            if testing_label_batches is not None:
                current_test_label_batch = testing_label_batches[i]
                print('number of points in this test window:', test_window.shape[0])
                print('percentage of class 1 points in this test window:',
                      100 * np.sum(current_test_label_batch) / test_window.shape[0])
            num_ref_drifts = 0 # how many training batches signal drift against this testing batch
            for j, ref_window in enumerate(reference_data_batches):
                print('\n REFERENCE BATCH #', j, 'of', len(reference_data_batches))
                if reference_label_batches is not None:
                    current_ref_label_batch = reference_label_batches[j]
                    print('number of points in this ref window:', ref_window.shape[0])
                    print('percentage of class 1 points in this ref window:',
                          100 * np.sum(current_ref_label_batch) / ref_window.shape[0])
                    drift_here, cluster_predict_acc = concept_drift_detected(
                        ref_window, test_window, additional_check, n_init, max_iter, tol, random_state,
                        reference_label_batch=reference_label_batches[j],
                        testing_label_batch=testing_label_batches[i],
                        debug=debug
                    )
                    print('drift_here')
                    print(drift_here)
                else:
                    drift_here, cluster_predict_acc = concept_drift_detected(
                        ref_window, test_window, additional_check, n_init, max_iter, tol, random_state,
                        debug=debug
                    )
                if drift_here:
                    num_ref_drifts += 1
                print('drift:', drift_here)

            drift = (num_ref_drifts / len(reference_data_batches)) > min_ref_batches_drift
            drifts_detected.append(drift)

    return drifts_detected


def get_final_drifts_from_all_info(drifts_2d_arr, len_ref_data_batches, min_ref_batches_drift):
    num_signals_each_testing_batch = np.sum(drifts_2d_arr, axis=0)
    drifts_detected = ((num_signals_each_testing_batch / len_ref_data_batches) > min_ref_batches_drift).tolist()
    return drifts_detected


def all_drifting_batches_parallel(
        reference_data_batches,
        testing_data_batches,
        min_ref_batches_drift,
        additional_check,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        reference_label_batches=None,
        testing_label_batches=None
):
    """
    Find all drift locations based on the given reference and testing batches

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param min_ref_batches_drift: the minimum fraction of reference batches that must signal drift for one test batch
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
    drifts_2d_arr, cluster_classif_accs_2d_arr = all_drifting_batches_parallel_all_info(
        reference_data_batches,
        testing_data_batches,
        additional_check,
        n_init,
        max_iter,
        tol,
        random_state,
        reference_label_batches,
        testing_label_batches
    )
    drifts_detected = get_final_drifts_from_all_info(drifts_2d_arr, len(reference_data_batches), min_ref_batches_drift)
    return drifts_detected


def all_drifting_batches_parallel_all_info(
        reference_data_batches,
        testing_data_batches,
        additional_check,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        reference_label_batches=None,
        testing_label_batches=None
):
    """
    Find all drift locations based on the given reference and testing batches

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param min_ref_batches_drift: the minimum fraction of reference batches that must signal drift for one test batch
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
    threshold = 0.05
    debug = False

    pool_iterables = []

    for i, test_window in enumerate(testing_data_batches):
        testing_label_batch = None
        if testing_label_batches is not None:
            testing_label_batch = testing_label_batches[i]
        for j, ref_window in enumerate(reference_data_batches):
            reference_label_batch = None
            if reference_label_batches is not None:
                reference_label_batch = reference_label_batches[j]
            entry = (ref_window, test_window,
                     additional_check,
                     n_init, max_iter, tol,
                     random_state,
                     threshold,
                     debug,
                     reference_label_batch,
                     testing_label_batch)
            pool_iterables.append(entry)

    # drifts_and_cluster_classif_acc_1d = []

    # drifts_and_cluster_classif_acc_1d = itertools.starmap(concept_drift_detected, pool_iterables)

    with Pool() as pool:
        print('pool opened')
        drifts_and_cluster_classif_acc_1d = pool.starmap(concept_drift_detected, pool_iterables)

    drifts_1d_tuple, cluster_classif_accs_1d_tuple = tuple(zip(*drifts_and_cluster_classif_acc_1d))
    drifts_1d_arr = np.asarray(drifts_1d_tuple)
    cluster_classif_accs_1d_arr = np.asarray(cluster_classif_accs_1d_tuple)

    # print('drifts_1d_arr')
    # print(drifts_1d_arr)
    # print('cluster_classif_accs_1d_arr')
    # print(cluster_classif_accs_1d_arr)

    drifts_2d_arr = drifts_1d_arr.reshape((len(testing_data_batches), len(reference_data_batches))).T
    cluster_classif_accs_2d_arr = cluster_classif_accs_1d_arr.reshape(
        (len(testing_data_batches), len(reference_data_batches))).T

    # num_signals_each_testing_batch = np.sum(drifts_2d_arr, axis=0)
    # drifts_detected = ((num_signals_each_testing_batch / len(testing_data_batches)) > min_ref_batches_drift).tolist()

    return drifts_2d_arr, cluster_classif_accs_2d_arr

