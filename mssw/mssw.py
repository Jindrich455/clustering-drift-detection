# TODOs:
# Unless specified otherwise, functions in this file work with numpy arrays
# The terms "benchmark data" and "reference data" mean the same thing, default is "reference data"
# The terms "slide data" and "testing data" mean the same thing, default is "testing data"

import numpy as np


# - function to calculate Ptg = the proportion of one feature attribute in one data point
#   --> for efficiency and elegance reasons: get this proportion directly for all benchmark data
def ptg_for_all(reference_data):
    # get one row as the sum of each column
    # divide each reference data row by this row
    # return ptgs, an array of same shape as benchmark data where each column should sum up to 1
    pass


# - function to calculate the information utility value of one feature attribute in reference (benchmark) data
#   --> for efficiency and elegance reasons: get the entropy and information utility directly for all attributes
def information_utilities_for_all(ptgs):
    # use scipy.stats.entropy, divide this row vector by ln(# rows in ptgs)
    # 1 - the resulting row vector to get a new vector of information utilities of each attribute
    # return a 2d array of the information utility of each attribute
    pass


# - function to calculate the weights of attributes from information utilities
def attribute_weights_for_all(information_utilities):
    # divide each information utility by the sum of all information utilities
    # return a 2d array of attribute weights
    pass


# - function to calculate the weights of attributes in the reference (benchmark) data
def get_attribute_weights_from(reference_data):
    ptgs = ptg_for_all(reference_data)
    information_utilities = information_utilities_for_all(ptgs)
    attribute_weights = attribute_weights_for_all(information_utilities)
    return attribute_weights


# - function to transform data by (the sqrt of) attribute weights
def transform_data_by_attribute_weights(original_data, attribute_weights):
    # square root attribute_weights
    # multiply each entry in original_data by each entry in sqrt_attribute_weights
    # return the weighted data
    pass


# function to transform multiple batches of data by (the sqrt of) attribute weights
def transform_batches_by_attribute_weights(original_data_batches, attribute_weights):
    # for each batch, call transform_data_by_attribute_weigths
    # return weighted batches
    pass


# - function to calculate JSEE, Av_ci for all i, and Av_sr for the given weighted input data
def calculate_clustering_statistics(weighted_sub_window, fitted_kmeans, num_clusters):
    # cluster the weighted input data through the fitted_kmeans object
    # for each weighted data point, calculate the Euclidean distance to its centroid (found by label)
    # --> store the sum of distances for each centroid in array1
    # --> store the number of points in each cluster in array2
    # JSEE = sum of all entries in array1
    # Av_ci = array1 entry-wise divided by array2
    # Av_sr = JSEE / length of weighted_sub_window
    pass


# - function to get Ss = the total average distance sequence of sub-windows in reference (benchmark) data
def get_s_s(weighted_reference_sub_windows, fitted_kmeans, num_clusters):
    # reference_sub_windows is an array where each entry is a sub-window
    # for each sub-window in sub_windows, apply calculate_three_statistics
    # return the array of Av_sr for each reference sub-window
    pass


# - function to calculate moving ranges MRi for each sub-window from Ss
def get_moving_ranges(s_s):
    # return the mean of an array of moving ranges obtained through numpy operations
    pass


def get_mean_s_s_and_mean_moving_ranges(weighted_reference_sub_windows, fitted_kmeans, num_clusters):
    # get s_s through get_s_s
    # get moving ranges with s_s
    # return the mean of both
    pass


# - function to test for concept drift based on the total average distance from one testing (slide) sub-window
def concept_drift_detected(mean_av_s, mean_mr, weighted_testing_sub_window, coeff=2.66):
    # check that the testing sub-window's Av_sr is within the boundaries set by mean_av_s, mean_mr, coeff
    # return True or False
    pass


def mssw_preprocess(refrence_data_batches, testing_data_batches):
    # join all reference data
    # fit minmaxscaler on <joined reference data>
    # scale <joined reference data>
    # scale all reference_data_batches
    # scale all testing_data_batches
    # obtain attribute weights: get_attribute_weights_from(<joined reference data>)
    # weigh the <joined reference data>
    # weigh all reference_data_batches and all testing_data_batches
    # return the weighted joined data, weighted reference batches and weighted testing batches
    pass


# - function to give all batches with concept drift from input data
def all_drifting_batches(reference_data_batches, testing_data_batches, num_clusters, coeff=2.66):
    # the batches accepted as input should be lists of numpy arrays

    # obtain scaled and weighted joined reference data and batches through mssw_preprocessing
    # use sklearn's kmeans to obtain clusters in <weighted joined reference data>
    # get the mean_av_s and mean_mr through get_s_s() and get_moving_ranges()
    # for each testing batch, run concept_drift_detected() and save the result in a list
    # return a list of boolean results of concept drift detections in all testing batches
    pass
