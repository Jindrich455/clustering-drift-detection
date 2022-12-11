# TODOs:
# Unless specified otherwise, functions in this file work with numpy arrays
# The terms "benchmark data" and "reference data" mean the same thing, default is "reference data"
# The terms "slide data" and "testing data" mean the same thing, default is "testing data"

# - function to calculate Ptg = the proportion of one feature attribute in one data point
#   --> for efficiency and elegance reasons: get this proportion directly for all benchmark data
import numpy as np


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


# - function to calculate the entropy-weighted distance between two data points
#   --> possibly input this as a lambda function directly to pyclustering for higher execution speed
def entropy_weighted_euclidean_distance(p1, p2, attribute_weights):
    distance = np.sqrt(np.multiply(np.square(np.subtract(p1, p2))))
    return distance


# - function to perform k-means on the reference (benchmark) data
def k_means_special_distance(reference_data, attribute_weights, num_clusters):
    # use pyclustering with a custom distance metric
    # return the fitted kmeans object
    pass


# - function to calculate JSEE = the sum of distances from each data point to its corresponding centroid
def calculate_jsee(data, fitted_kmeans, attribute_weights):
    # predict the input data through the fitted_kmeans object
    # for each data point, calculate the weighted distance to its centroid
    # return the sum of these weighted distances
    pass


# - function to calculate Av_ci = the average distance from all points in cluster ci to their centroid
def calculate_av_ci(data, fitted_kmeans, attribute_weights):
    # for each row in data, get the distances to each centroid (entropy-weighted)
    # return an array of av_ci with as many entries as there are centroids
    pass


# - function to calculate Av_sr = the total average distance of points from clusters in one (sub-)window
def calculate_av_sr(sub_window, fitted_kmeans, attribute_weights):
    # get the jsee for the sub-window
    # divide this by the number of samples in this sub-window
    # return the av_sr
    pass


# - function to get Ss = the total average distance sequence of sub-windows in reference (benchmark) data
def get_s_s(reference_sub_windows, fitted_kmeans, attribute_weights):
    # sub_windows is an array where each entry is a sub-window
    # for each sub-window in sub_windows, apply calculate_av_sr
    # return the array of Av_sr for each sub-window
    pass


# - function to calculate moving ranges MRi for each sub-window from Ss
def get_moving_ranges(s_s):
    # return an array of moving ranges through numpy operations
    pass


# - function to calculate the total average distance of a testing (slide) sub-window
def get_testing_av_sr(testing_sub_window, fitted_kmeans):
    # use calculate_av_sr
    pass


# - function to test for concept drift based on the total average distance from one testing (slide) sub-window
def concept_drift_detected(mean_av_s, mean_mr, testing_sub_window, coeff=2.66):
    # check that testing_av_sr is within the boundaries set by mean_av_s, mean_mr, coeff
    # return True or False
    pass


# - function to give all batches with concept drift from input data
def all_drifting_batches(reference_data_batches, testing_data_batches, num_clusters):
    # the batches accepted as input should be numpy arrays of numpy arrays

    # join all reference data
    # obtain attribute weights: get_attribute_weights_from(<joined reference data>)
    # and use k_means_special_distance() on the joined reference data to obtain clusters
    # get the mean_av_s and mean_mr through get_s_s() and get_moving_ranges()
    # for each testing batch, run concept_drift_detected() and save the result in a list
    # return a list of boolean results of concept drift detections in all testing batches
    pass
