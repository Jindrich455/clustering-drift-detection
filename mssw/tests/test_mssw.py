from unittest import TestCase
import numpy.testing

import numpy as np

import mssw.mssw


class TestMSSW(TestCase):
    def test_ptg_for_all(self):
        reference_data = np.array([[3, 7], [3, 6]])
        ptg = mssw.mssw.ptg_for_all(reference_data)
        correct_ptg = np.array([[0.5, 7/13], [0.5, 6/13]])
        np.testing.assert_array_equal(ptg, correct_ptg)

    def test_information_utilities_for_all(self):
        ptgs = np.array([[3, 7], [3, 6], [1, 4]])
        inf_util = mssw.mssw.information_utilities_for_all(ptgs)
        correct_inf_util_shape = np.array([[-1, -1]]).shape
        np.testing.assert_equal(correct_inf_util_shape, inf_util.shape)

    def test_attribute_weights_for_all(self):
        information_utilities = np.array([[3, 6, 1]])
        attribute_weights = mssw.mssw.attribute_weights_for_all(information_utilities)
        correct_attribute_weights = np.array([[0.3, 0.6, 0.1]])
        np.testing.assert_array_equal(attribute_weights, correct_attribute_weights)

    def test_get_attribute_weights_from(self):
        reference_data = np.array([[3, 7], [3, 6]])
        attribute_weights = mssw.mssw.get_attribute_weights_from(reference_data)
        correct_shape_attribute_weights = np.array([[-1, -1]]).shape
        np.testing.assert_equal(attribute_weights.shape, correct_shape_attribute_weights)

    def test_transform_data_by_attribute_weights(self):
        original_data = np.array([[3, 7], [4, 5], [2, 4]])
        attribute_weights = np.array([[4, 9]])
        weighted_data = mssw.mssw.transform_data_by_attribute_weights(original_data, attribute_weights)
        correct_weighted_data = np.array([[6, 21], [8, 15], [4, 12]])
        np.testing.assert_array_equal(weighted_data, correct_weighted_data)

    def test_transform_batches_by_attribute_weights(self):
        original_batches = [np.array([[1, 3, 4], [3, 7, 8]]), np.array([[3, 2, 5], [2, 0, 9]])]
        dummy_weights = np.array([[0.6, 0.3, 0.1]])
        weighted_batches = mssw.mssw.transform_batches_by_attribute_weights(original_batches, dummy_weights)
        for batch in weighted_batches:
            np.testing.assert_equal(batch.shape, (2, 3))

    # def test_calculate_clustering_statistics(self):
    #     np.testing.assert_equal(True, False)

    def test_get_s_s(self):
        np.testing.assert_equal(True, False)

    def test_get_moving_ranges(self):
        np.testing.assert_equal(True, False)

    def test_get_mean_s_s_and_mean_moving_ranges(self):
        np.testing.assert_equal(True, False)
