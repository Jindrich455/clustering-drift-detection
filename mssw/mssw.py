# TODOs:
# Unless specified otherwise, functions in this file work with numpy arrays
# The terms "benchmark data" and "reference data" mean the same thing, default is "reference data"
# The terms "slide data" and "testing data" mean the same thing, default is "testing data"

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# - function to calculate Ptg = the proportion of one feature attribute in one data point
#   --> for efficiency and elegance reasons: get this proportion directly for all benchmark data
import scipy.stats
import sklearn.metrics.pairwise


def ptg_for_all(reference_data):
    # get one row as the sum of each column
    column_sum = np.sum(reference_data, axis=0)
    # divide each reference data row by this row
    # return ptgs, an array of same shape as benchmark data where each column should sum up to 1
    return np.divide(reference_data, column_sum)


# - function to calculate the information utility value of one feature attribute in reference (benchmark) data
#   --> for efficiency and elegance reasons: get the entropy and information utility directly for all attributes
def information_utilities_for_all(ptgs):
    # use scipy.stats.entropy, divide this row vector by ln(# rows in ptgs)
    entropies = np.divide(scipy.stats.entropy(ptgs, axis=0), ptgs.shape[0])
    # 1 - the resulting row vector to get a new vector of information utilities of each attribute
    information_utilities = np.subtract(1, entropies).reshape((1, entropies.shape[0]))
    # return a 2d array of the information utility of each attribute
    return information_utilities


# - function to calculate the weights of attributes from information utilities
def attribute_weights_for_all(information_utilities):
    # divide each information utility by the sum of all information utilities
    attribute_weights = np.divide(information_utilities, np.sum(information_utilities))
    # return a 2d array of attribute weights
    return attribute_weights


# - function to calculate the weights of attributes in the reference (benchmark) data
def get_attribute_weights_from(reference_data):
    ptgs = ptg_for_all(reference_data)
    information_utilities = information_utilities_for_all(ptgs)
    attribute_weights = attribute_weights_for_all(information_utilities)
    return attribute_weights


# - function to transform data by (the sqrt of) attribute weights
def transform_data_by_attribute_weights(original_data, attribute_weights):
    # square root attribute_weights
    sqrt_attribute_weights = np.sqrt(attribute_weights)
    # multiply each entry in original_data by each entry in sqrt_attribute_weights
    weighted_data = np.multiply(original_data, sqrt_attribute_weights)
    # return the weighted data
    return weighted_data


# function to transform multiple batches of data by (the sqrt of) attribute weights
def transform_batches_by_attribute_weights(original_batches, attribute_weights):
    # for each batch, call transform_data_by_attribute_weigths
    weighted_batches = []
    for original_batch in original_batches:
        weighted_batches.append(transform_data_by_attribute_weights(original_batch, attribute_weights))
    # return weighted batches
    return weighted_batches


# - function to calculate JSEE, Av_ci for all i, and Av_sr for the given weighted input data
def calculate_clustering_statistics(weighted_sub_window, fitted_kmeans, num_clusters):
    # cluster the weighted input data through the fitted_kmeans object
    centroids = fitted_kmeans.cluster_centers_
    predicted_cluster_labels = fitted_kmeans.predict(weighted_sub_window)
    sum_centroid_distances = np.zeros(num_clusters).reshape((1, num_clusters))
    num_points_in_clusters = np.zeros(num_clusters).reshape((1, num_clusters))
    # for each weighted data point, calculate the Euclidean distance to its centroid (found by label)
    for i, label in enumerate(predicted_cluster_labels):
        num_points_in_clusters[0, label] += 1
        # the following numpy operations are a simple Euclidean distance
        sum_centroid_distances[0, label] += np.linalg.norm(np.subtract(centroids[label], weighted_sub_window[i]))
    # --> store the sum of distances for each centroid in array1
    # --> store the number of points in each cluster in array2
    # JSEE = sum of all entries in array1
    JSEE = np.sum(sum_centroid_distances)
    # Av_c = array1 entry-wise divided by array2
    Av_c = np.divide(sum_centroid_distances, num_points_in_clusters)
    # Av_sr = JSEE / length of weighted_sub_window
    Av_sr = JSEE / weighted_sub_window.shape[0]
    return JSEE, Av_c, Av_sr


# - function to get Ss = the total average distance sequence of sub-windows in reference (benchmark) data
def get_s_s(weighted_reference_sub_windows, fitted_kmeans, num_clusters):
    # weighted_reference_sub_windows is an array where each entry is a sub-window
    num_sub_windows = len(weighted_reference_sub_windows)
    s_s = np.zeros(num_sub_windows).reshape((1, num_sub_windows))
    # for each sub-window in sub_windows, apply calculate_three_statistics
    for i, weighted_reference_sub_window in enumerate(weighted_reference_sub_windows):
        _, _, Av_sr = calculate_clustering_statistics(weighted_reference_sub_window, fitted_kmeans, num_clusters)
        s_s[0, i] = Av_sr
    # return the array of Av_sr for each reference sub-window
    return s_s


# - function to calculate moving ranges MRi for each sub-window from Ss
def get_moving_ranges(s_s):
    # return the moving ranges obtained through numpy operations
    moving_ranges = np.abs(np.subtract(s_s[:, 1:], s_s[:, :-1]))
    return moving_ranges


def get_mean_s_s_and_mean_moving_ranges(weighted_reference_sub_windows, fitted_kmeans, num_clusters):
    # get s_s through get_s_s
    s_s = get_s_s(weighted_reference_sub_windows, fitted_kmeans, num_clusters)
    # get moving ranges with s_s
    moving_ranges = get_moving_ranges(s_s)
    # return the mean of both
    return np.mean(s_s), np.mean(moving_ranges)


# - function to test for concept drift based on the total average distance from one testing (slide) sub-window
def concept_drift_detected(mean_av_s, mean_mr, weighted_testing_sub_window, fitted_kmeans, num_clusters, coeff):
    # check that the testing sub-window's Av_sr is within the boundaries set by mean_av_s, mean_mr, coeff
    UCL_Av_s = mean_av_s + coeff * mean_mr
    LCL_Av_s = mean_av_s - coeff * mean_mr
    _, _, test_Av_sr = calculate_clustering_statistics(weighted_testing_sub_window, fitted_kmeans, num_clusters)

    return not (LCL_Av_s < test_Av_sr < UCL_Av_s)


def mssw_preprocess(reference_data_batches, testing_data_batches):
    # join all reference data
    joined_reference_data = reference_data_batches[0]
    for reference_batch in reference_data_batches[1:]:
        np.append(joined_reference_data, reference_batch, axis=0)
    # fit minmaxscaler on <joined reference data>
    scaler = MinMaxScaler()
    scaler.fit(joined_reference_data)
    # scale <joined reference data>
    joined_reference_data = scaler.transform(joined_reference_data)
    # scale all reference_data_batches
    reference_data_batches = [scaler.transform(batch) for batch in reference_data_batches]
    # scale all testing_data_batches
    testing_data_batches = [scaler.transform(batch) for batch in testing_data_batches]
    # obtain attribute weights: get_attribute_weights_from(<joined reference data>)
    attribute_weights = get_attribute_weights_from(joined_reference_data)
    # weigh the <joined reference data>
    weighted_joined_reference_data = transform_data_by_attribute_weights(joined_reference_data, attribute_weights)
    # weigh all reference_data_batches and all testing_data_batches
    weighted_reference_batches =\
        [transform_data_by_attribute_weights(batch, attribute_weights) for batch in reference_data_batches]
    weighted_testing_batches =\
        [transform_data_by_attribute_weights(batch, attribute_weights) for batch in testing_data_batches]
    # return the weighted joined data, weighted reference batches and weighted testing batches
    return weighted_joined_reference_data, weighted_reference_batches, weighted_testing_batches


# - function to give all batches with concept drift from input data
def all_drifting_batches(reference_data_batches, testing_data_batches, num_clusters, random_state=0, coeff=2.66):
    # the batches accepted as input should be lists of numpy arrays

    # obtain scaled and weighted joined reference data and batches through mssw_preprocessing
    weighted_joined_reference_data, weighted_reference_batches, weighted_testing_batches =\
        mssw_preprocess(reference_data_batches, testing_data_batches)
    # use sklearn's kmeans to obtain clusters in <weighted joined reference data>
    fitted_kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(weighted_joined_reference_data)
    # get the mean_av_s and mean_mr through get_mean_s_s_and_mean_moving_ranges
    mean_av_s, mean_mr = get_mean_s_s_and_mean_moving_ranges(weighted_reference_batches, fitted_kmeans, num_clusters)
    # for each testing batch, run concept_drift_detected() and save the result in a list
    drifts_detected = []
    for weighted_testing_batch in weighted_testing_batches:
        drifts_detected.append(concept_drift_detected(
            mean_av_s, mean_mr, weighted_testing_batch, fitted_kmeans, num_clusters, coeff))
    # return a list of boolean results of concept drift detections in all testing batches
    pass
