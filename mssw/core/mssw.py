"""
Drift detection algorithm from
[1] Y. Yuan, Z. Wang, and W. Wang,
“Unsupervised concept drift detection based on multi-scale slide windows,”
Ad Hoc Networks, vol. 111, p. 102325, Feb. 2021, doi: 10.1016/j.adhoc.2020.102325.

MSSW is an abbreviation for Multi-Scale Sliding Windows

- Unless specified otherwise, functions in this file work with numpy arrays
- The terms "benchmark data" and "reference data" mean the same thing, default is "reference data"
- The terms "slide data" and "testing data" mean the same thing, default is "testing data"
"""
import numpy as np
from sklearn.cluster import KMeans
# import mssw.mssw_preprocessing
from . import mssw_preprocessing
# from mssw import mssw_preprocessing


def obtain_cluster_distances_and_sizes(weighted_sub_window, fitted_kmeans, n_clusters):
    """
    Get the sum of centroid distances and size for clusters formed by fitted_kmeans and weighted_sub_window

    :param weighted_sub_window: array of shape (#points, #attributes) of weighted data
    :param fitted_kmeans: fitted sklearn kmeans object to use for clustering of the weighted_sub_window
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: (array of shape (1, n_clusters) of sums of centroid distances,
    array of shape (1, n_clusters) of cluster sizes)
    """
    centroids = fitted_kmeans.cluster_centers_
    predicted_cluster_labels = fitted_kmeans.predict(weighted_sub_window)

    centroid_distance_sums = np.zeros(n_clusters).reshape((1, n_clusters))
    num_points_in_clusters = np.zeros(n_clusters).reshape((1, n_clusters))
    for cluster_id in range(n_clusters):
        cluster_mask = predicted_cluster_labels == cluster_id
        cluster = weighted_sub_window[cluster_mask]

        num_points_in_clusters[0, cluster_id] = cluster.shape[0]

        centroid = centroids[cluster_id]
        centroid_diffs = np.subtract(cluster, centroid)
        euclideans = np.linalg.norm(centroid_diffs, axis=1)
        sum_euclideans = np.sum(euclideans)
        centroid_distance_sums[0, cluster_id] = sum_euclideans

    return centroid_distance_sums, num_points_in_clusters


def calculate_clustering_statistics(weighted_sub_window, fitted_kmeans, n_clusters):
    """
    Cluster the given weighted_sub_window, and then obtain JSEE, Av_ci for all i, and Av_sr from it

    :param weighted_sub_window: array of shape (#points, #attributes) of weighted data
    :param fitted_kmeans: fitted sklearn kmeans object to use for clustering of the weighted_sub_window
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: (JSEE float, Av_ci array of shape (1, #attributes), Av_sr float)
    """
    centroid_distance_sums, num_points_in_clusters = obtain_cluster_distances_and_sizes(
        weighted_sub_window, fitted_kmeans, n_clusters
    )

    JSEE = np.sum(centroid_distance_sums)
    print('centroid distance sums', centroid_distance_sums)
    print('num points in clusters', num_points_in_clusters)
    Av_c = np.divide(centroid_distance_sums, num_points_in_clusters)
    Av_sr = JSEE / weighted_sub_window.shape[0]
    return JSEE, Av_c, Av_sr


def get_s_s(weighted_reference_sub_windows, fitted_kmeans, n_clusters):
    """
    Get S_s = the total average distance sequence of sub-windows in reference (benchmark) data

    :param weighted_reference_sub_windows: list of arrays of shape (n_r, #attributes) of weighted reference data,
        r is the sub-window number, n_r=#points in this sub-window
    :param fitted_kmeans: sklearn kmeans object previously fitted to weighted reference (benchmark) data
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: array of shape (1, len(weighted_reference_sub_windows))
    """
    num_sub_windows = len(weighted_reference_sub_windows)
    s_s = np.zeros(num_sub_windows).reshape((1, num_sub_windows))
    for i, weighted_reference_sub_window in enumerate(weighted_reference_sub_windows):
        _, _, Av_sr = calculate_clustering_statistics(weighted_reference_sub_window, fitted_kmeans, n_clusters)
        s_s[0, i] = Av_sr
    return s_s


def get_moving_ranges(s_s):
    """
    Get moving ranges (MR_i) for each sub-window from S_s

    :param s_s: s_s as obtained from get_s_s(...)
    :return: array of shape (1, len(s_s)-1)
    """
    moving_ranges = np.abs(np.subtract(s_s[:, 1:], s_s[:, :-1]))
    return moving_ranges


def get_mean_s_s_and_mean_moving_ranges(weighted_reference_sub_windows, fitted_kmeans, n_clusters):
    """
    Find the S_s and MR sequences and return their mean

    :param weighted_reference_sub_windows: list of arrays of shape (n_r, #attributes) of weighted reference data,
        r is the sub-window number, n_r=#points in this sub-window
    :param fitted_kmeans: sklearn kmeans object previously fitted to weighted reference (benchmark) data
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: (mean of S_s as float, mean of MR as float)
    """
    s_s = get_s_s(weighted_reference_sub_windows, fitted_kmeans, n_clusters)
    moving_ranges = get_moving_ranges(s_s)
    return np.mean(s_s), np.mean(moving_ranges)


# - function to test for concept drift based on the total average distance from one testing (slide) sub-window
def concept_drift_detected(mean_av_s, mean_mr, weighted_testing_sub_window, fitted_kmeans, n_clusters, coeff):
    """
    Test for concept drift in one weighted testing sub-window

    :param mean_av_s: mean_s_s as obtained from get_mean_s_s_and_mean_moving_ranges(...)
    :param mean_mr: mean_mr as obtained from get_mean_s_s_and_mean_moving_ranges(...)
    :param weighted_testing_sub_window: array of shape (#points, #attributes) of one weighted testing sub-window
    :param fitted_kmeans: sklearn kmeans object previously fitted to weighted reference (benchmark) data
    :param n_clusters: number of clusters used to fit the kmeans object
    :param coeff: drift detection coefficient
    :return: True if drift is detected, False otherwise
    """
    UCL_Av_s = mean_av_s + coeff * mean_mr
    LCL_Av_s = mean_av_s - coeff * mean_mr
    _, _, test_Av_sr = calculate_clustering_statistics(weighted_testing_sub_window, fitted_kmeans, n_clusters)

    return not (LCL_Av_s < test_Av_sr < UCL_Av_s)


def all_drifting_batches(
        reference_data_batches,
        testing_data_batches,
        n_clusters=2,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        coeff=2.66
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
    weighted_joined_reference_data, weighted_reference_batches, weighted_testing_batches =\
        mssw_preprocessing.mssw_preprocess(reference_data_batches, testing_data_batches)

    fitted_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    ).fit(weighted_joined_reference_data)
    mean_av_s, mean_mr = get_mean_s_s_and_mean_moving_ranges(weighted_reference_batches, fitted_kmeans, n_clusters)

    drifts_detected = []
    for weighted_testing_batch in weighted_testing_batches:
        drifts_detected.append(concept_drift_detected(
            mean_av_s, mean_mr, weighted_testing_batch, fitted_kmeans, n_clusters, coeff))
    return drifts_detected
