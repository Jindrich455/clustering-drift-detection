import unittest

import accepting
import preprocessing
import ucdd
from sklearn.preprocessing import MinMaxScaler


def drift_occurrences_shortcut(dataset_path, test_fraction, num_ref_batches, num_test_batches, scaling,
                               scaler, debug=False, random_state=0):
    df_x, df_y = accepting.get_clean_df(dataset_path)
    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.prepare_data_and_get_batches(
        df_x, df_y, test_fraction, num_ref_batches, num_test_batches,
        scaling, scaler, use_categorical=False, encoding=False, encoder=None
    )

    return ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state)


class TestUCDD(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_failing_on_purpose(self):
        self.assertEqual(True, False)  # add assertion here

    def test_something_else(self):
        self.assertEqual(10, 10)  # add assertion here

    def test_drift_one_testing_batch(self):
        drift_occurrences = drift_occurrences_shortcut(
            'test_datasets/drift_2d.arff', test_fraction=0.5,
            num_ref_batches=1, num_test_batches=1, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([0], drift_occurrences)

    def test_no_drift_one_testing_batch(self):
        drift_occurrences = drift_occurrences_shortcut(
            'test_datasets/no_drift_2d.arff', test_fraction=0.5,
            num_ref_batches=1, num_test_batches=1, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([], drift_occurrences)

    def test_drift_three_testing_batches(self):
        drift_occurrences = drift_occurrences_shortcut(
            'test_datasets/drift_from_p21_2d.arff', test_fraction=0.75,
            num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([1, 2], drift_occurrences)

    def test_drift_three_testing_batches_other_class(self):
        drift_occurrences = drift_occurrences_shortcut(
            'test_datasets/drift_from_p21_2d_other_class.arff', test_fraction=0.75,
            num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([1, 2], drift_occurrences)

    def test_drift_three_testing_batches_more_neighbours(self):
        drift_occurrences = drift_occurrences_shortcut(
            'test_datasets/drift_from_p21_2d_more_neighbours.arff', test_fraction=0.75,
            num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([1, 2], drift_occurrences)


if __name__ == '__main__':
    unittest.main()
