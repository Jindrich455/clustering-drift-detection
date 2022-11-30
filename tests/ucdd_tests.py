import unittest
import preprocessing
import ucdd
from sklearn.preprocessing import MinMaxScaler


class TestUCDD(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_failing_on_purpose(self):
        self.assertEqual(True, False)  # add assertion here

    def test_something_else(self):
        self.assertEqual(10, 10)  # add assertion here

    def drift_occurrences_shortcut(self, dataset_path, test_fraction, num_ref_batches, num_test_batches, scaling,
                                   scaler, debug=False, random_state=0):
        X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.get_batches(
            dataset_path, test_fraction,
            num_ref_batches, num_test_batches, scaling, scaler, debug)
        return ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state)

    def test_drift_one_testing_batch(self):
        drift_occurrences = self.drift_occurrences_shortcut(
            'test_datasets/drift_2d.arff', test_fraction=0.5,
            num_ref_batches=1, num_test_batches=1, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([0], drift_occurrences)

    def test_no_drift_one_testing_batch(self):
        drift_occurrences = self.drift_occurrences_shortcut(
            'test_datasets/no_drift_2d.arff', test_fraction=0.5,
            num_ref_batches=1, num_test_batches=1, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([], drift_occurrences)

    def test_drift_three_testing_batches(self):
        drift_occurrences = self.drift_occurrences_shortcut(
            'test_datasets/drift_from_p21_2d.arff', test_fraction=0.75,
            num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([1, 2], drift_occurrences)

    def test_drift_three_testing_batches_other_class(self):
        drift_occurrences = self.drift_occurrences_shortcut(
            'test_datasets/drift_from_p21_2d_other_class.arff', test_fraction=0.75,
            num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([1, 2], drift_occurrences)

    def test_drift_three_testing_batches_more_neighbours(self):
        drift_occurrences = self.drift_occurrences_shortcut(
            'test_datasets/drift_from_p21_2d_more_neighbours.arff', test_fraction=0.75,
            num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler())
        self.assertEqual([1, 2], drift_occurrences)


if __name__ == '__main__':
    unittest.main()
