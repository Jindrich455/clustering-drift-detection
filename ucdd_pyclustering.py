"""
Implementation of the UCDD algorithm by Dan Shang, Guangquan Zhang and Jie Lu
"""
import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import scipy
from pyclustering.utils import distance_metric, type_metric
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import supported_parameters as spms


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


def pyclusters_metric_from_id(metric_id):
    converter = {
        spms.Distances.EUCLIDEAN: distance_metric(type_metric.EUCLIDEAN),
        spms.Distances.EUCLIDEAN_SQUARE: distance_metric(type_metric.EUCLIDEAN_SQUARE),
        spms.Distances.MANHATTAN: distance_metric(type_metric.MANHATTAN),
        spms.Distances.CHEBYSHEV: distance_metric(type_metric.CHEBYSHEV),
        spms.Distances.CANBERRA: distance_metric(type_metric.CANBERRA)
    }
    return converter[metric_id]


def join_predict_split(df_ref_window, df_test_window, random_state, metric_id):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    # join the points from two windows
    df_union = pd.concat([df_ref_window, df_test_window])
    df_union_new_index = df_union.set_index(np.arange(len(df_union)))

    # predict their label values
    # predicted_labels = KMeans(n_clusters=2, random_state=random_state).fit_predict(df_union_new_index)

    # Prepare initial centers using K-Means++ method.
    # metric = distance_metric(type_metric.USER_DEFINED, func=scipy.spatial.distance.euclidean)
    # metric = distance_metric(type_metric.EUCLIDEAN)

    metric = pyclusters_metric_from_id(metric_id)
    # metric = distance_metric(type_metric.CHEBYSHEV)

    initial_centers = kmeans_plusplus_initializer(df_union, 2, random_state=random_state).initialize()
    # Create instance of K-Means algorithm with prepared centers.
    kmeans_instance = kmeans(df_union, initial_centers, random_state=random_state, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    # print('clusters')
    # print(clusters)
    final_centers = kmeans_instance.get_centers()
    # print('final centers')
    # print(final_centers)

    predicted_labels = np.repeat(0, len(df_union))
    predicted_labels[clusters[0]] = 1
    # print('predicted_labels')
    # print(predicted_labels)

    plus_prediction = df_union.iloc[clusters[0]]
    # print('plus_prediction')
    # print(plus_prediction)

    minus_prediction = df_union.iloc[clusters[1]]
    # print('minus_prediction')
    # print(minus_prediction)

    # predict = kmeans_instance.predict(df_union)
    # print('predict')
    # print(predict)
    # # Visualize obtained results
    # kmeans_visualizer.show_clusters(df_union, clusters, final_centers)



    # split values by predicted label and window
    return split_back_to_windows(df_union_new_index, predicted_labels, len(df_ref_window), len(df_test_window))


def nn_metric_from_id(metric_id):
    converter = {
        spms.Distances.EUCLIDEAN: 'euclidean',
        spms.Distances.EUCLIDEAN_SQUARE: 'sqeuclidean',
        spms.Distances.MANHATTAN: 'manhattan',
        spms.Distances.CHEBYSHEV: 'chebyshev',
        spms.Distances.CANBERRA: 'canberra'
    }
    return converter[metric_id]


def compute_neighbors(u, v, metric_id, debug_string='v'):
    neigh = NearestNeighbors(n_neighbors=1, metric=nn_metric_from_id(metric_id))
    # neigh = NearestNeighbors(n_neighbors=1, metric='chebyshev')
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
    if debug: print('beta', beta)
    return beta


def detect_cd(df_X_ref, df_X_test, random_state, show_2d_plots, additional_check, debug, metric_id,
              threshold=0.05):
    """Detect whether a concept drift occurred based on a reference and a testing window"""
    df_ref_plus, df_ref_minus, df_test_plus, df_test_minus = \
        join_predict_split(df_X_ref, df_X_test, random_state, metric_id)

    if show_2d_plots:
        ucdd_plotter.plot_predicted(df_ref_plus, df_ref_minus, df_test_plus, df_test_minus)

    if debug: print('BETA MINUS (ref+, ref-, test-)')
    beta_minus = compute_beta(df_ref_plus, df_ref_minus, df_test_minus, show_2d_plots, debug, metric_id)
    if debug: print('BETA PLUS (ref-, ref+, test+)')
    beta_plus = compute_beta(df_ref_minus, df_ref_plus, df_test_plus, show_2d_plots, debug, metric_id)

    drift = False
    if beta_plus < threshold or beta_minus < threshold:
        drift = True
    if additional_check and ((1 - beta_plus) < threshold or (1 - beta_minus) < threshold):
        drift = True
    return drift


def drift_occurrences_list(
        X_ref_batches, X_test_batches, random_state, additional_check, detect_all_training_batches,
        show_2d_plots=False, debug=False, metric_id=spms.Distances.EUCLIDEAN
):
    """Return a list of all batches where the algorithm detected drift"""
    print('############################ USING PYCLUSTERING ############################')

    drift_signal_locations = []
    for i, df_X_test in enumerate(X_test_batches):
        print('#### TEST BATCH', i, 'of', len(X_test_batches), '####')
        drift = False
        if detect_all_training_batches:
            drift = True
        for j, df_X_ref in enumerate(X_ref_batches):
            if debug: print('--- training batch', j, '---')
            drift_here = detect_cd(df_X_ref, df_X_test, random_state, show_2d_plots, additional_check, debug,
                                   metric_id)
            if detect_all_training_batches:
                drift = drift & drift_here
            else:
                drift = drift | drift_here
        if drift:
            drift_signal_locations.append(i)
        if debug: print('\n\n')
    return drift_signal_locations
