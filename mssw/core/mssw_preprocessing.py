import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler


def ptg_for_all(reference_data):
    """
    Calculate all P_tgs from reference data

    :param reference_data: array of shape (#points, #attributes) of reference data
    :return: array of shape (#points, #attribute) of corresponding P_tgs
    """
    column_sum = np.sum(reference_data, axis=0)
    return np.divide(reference_data, column_sum)


def information_utilities_for_all(ptgs):
    """
    Calculate information utility values from P_tgs

    :param ptgs: P_tgs as obtained from ptg_for_all(...)
    :return: array of shape (1, #attributes) of the information utility of each attribute
    """
    entropies = np.divide(scipy.stats.entropy(ptgs, axis=0), np.log(ptgs.shape[0]))
    entropies = np.where(entropies > 1, 1.0, entropies)
    information_utilities = np.subtract(1, entropies).reshape((1, entropies.shape[0]))
    return information_utilities


def attribute_weights_for_all(information_utilities):
    """
    Calculate the weights of attributes from information utilities

    :param information_utilities: information utilities as obtained from information_utilities_for_all(...)
    :return: array of shape (1, #attributes) of the attribute weights of each attribute
    """
    attribute_weights = np.divide(information_utilities, np.sum(information_utilities))
    return attribute_weights


def get_attribute_weights_from(reference_data):
    """
    Calculate weights of attributes from reference (benchmark) data

    :param reference_data: array of shape (#points, #attributes)
    :return: array of shape (1, #attributes) of the attribute weights of each attribute
    """
    ptgs = ptg_for_all(reference_data)
    # print('# zero values in ptgs', np.count_nonzero(ptgs == 0))
    # print('# negative values in ptgs', np.count_nonzero(ptgs < 0))
    # print('# nan values in ptgs', np.count_nonzero(np.isnan(ptgs)))
    information_utilities = information_utilities_for_all(ptgs)
    # print('# zero values in information utilities', np.count_nonzero(information_utilities == 0))
    # print('# negative values in information utilities', np.count_nonzero(information_utilities < 0))
    # print('# nan values in information utilities', np.count_nonzero(np.isnan(information_utilities)))
    attribute_weights = attribute_weights_for_all(information_utilities)
    # print('# zero values in attribute weights', np.count_nonzero(attribute_weights == 0))
    # print('# negative values in attribute weights', np.count_nonzero(attribute_weights < 0))
    # print('# nan values in attribute weights', np.count_nonzero(np.isnan(attribute_weights)))
    return attribute_weights


def transform_data_by_attribute_weights(original_data, attribute_weights):
    """
    Transform data by the sqrt of attribute weights

    :param original_data: array of shape (#points, #attributes) to transform
    :param attribute_weights: array of shape (1, #attributes) to use for the transformation
    :return: array of shape (#points, #attributes) of weighted data
    """
    sqrt_attribute_weights = np.sqrt(attribute_weights)
    weighted_data = np.multiply(original_data, sqrt_attribute_weights)
    return weighted_data


def transform_batches_by_attribute_weights(original_batches, attribute_weights):
    """
    Transform multiple batches of data by the sqrt of attribute weights

    :param original_batches: list of arrays of shape (n_i, #attributes), i=batch number, n_i > 1
    :param attribute_weights: array of shape (1, #attributes) of weights to use for the transformation
    :return: list of arrays of shape(n_i, #attributes) of weighted data
    """
    weighted_batches = []
    for original_batch in original_batches:
        weighted_batches.append(transform_data_by_attribute_weights(original_batch, attribute_weights))
    return weighted_batches


def mssw_preprocess(reference_data_batches, testing_data_batches):
    """
    Preprocess data batches through minmax scaling, apply weighting so that Euclidean distance on this weighted data
    becomes the desired entropy-weighted distance on the original data

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :return: (array of shape (sum(n_r_r) #attributes) of joined reference data, weighted reference batches (same
        structure as reference_data_batches), weighted testing batches (same structure as testing_data_batches))
    """
    joined_reference_data = reference_data_batches[0]
    for reference_batch in reference_data_batches[1:]:
        np.append(joined_reference_data, reference_batch, axis=0)

    scaler = MinMaxScaler()
    scaler.fit(joined_reference_data)
    joined_reference_data = scaler.transform(joined_reference_data)
    reference_data_batches = [scaler.transform(batch) for batch in reference_data_batches]
    testing_data_batches = [scaler.transform(batch) for batch in testing_data_batches]

    # print('# zero values in scaled data', np.count_nonzero(joined_reference_data == 0))
    # print('# negative values in scaled data', np.count_nonzero(joined_reference_data < 0))
    # print('# nan values in scaled data', np.count_nonzero(np.isnan(joined_reference_data)))
    small_float = np.finfo(dtype=float).eps * (10 ** 6)
    joined_reference_data = np.where(joined_reference_data == 0, small_float, joined_reference_data)
    reference_data_batches = [np.where(batch == 0, small_float, batch) for batch in reference_data_batches]
    testing_data_batches = [np.where(batch == 0, small_float, batch) for batch in testing_data_batches]
    # print('after transformation')
    # print('# zero values in scaled data', np.count_nonzero(joined_reference_data == 0))
    # print('# negative values in scaled data', np.count_nonzero(joined_reference_data < 0))
    # print('# nan values in scaled data', np.count_nonzero(np.isnan(joined_reference_data)))

    attribute_weights = get_attribute_weights_from(joined_reference_data)
    # print('# negative values in attribute weights', np.count_nonzero(attribute_weights < 0))
    # print('# nan values in attribute weights', np.count_nonzero(np.isnan(attribute_weights)))
    weighted_joined_reference_data = transform_data_by_attribute_weights(joined_reference_data, attribute_weights)
    weighted_reference_batches =\
        [transform_data_by_attribute_weights(batch, attribute_weights) for batch in reference_data_batches]
    weighted_testing_batches =\
        [transform_data_by_attribute_weights(batch, attribute_weights) for batch in testing_data_batches]
    return weighted_joined_reference_data, weighted_reference_batches, weighted_testing_batches
