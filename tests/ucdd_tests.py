import unittest

from sklearn.compose import ColumnTransformer

import accepting
import my_preprocessing
import ucdd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_selector as selector

import ucdd_eval

random_state = 0
use_additional_check = True


class TestUCDD(unittest.TestCase):
    def test_failing_on_purpose(self):
        self.assertEqual(True, False)  # add assertion here

    def test_drift_one_testing_batch(self):
        drift_occurrences = ucdd_eval.evaluate_ucdd(
            file_path='test_datasets/drift_2d.arff',
            scaling="minmax",
            encoding="none",
            test_size=0.5,
            num_ref_batches=1,
            num_test_batches=1,
            random_state=random_state,
            additional_check=False
        )

        self.assertEqual([0], drift_occurrences)

    def test_no_drift_one_testing_batch(self):
        drift_occurrences = ucdd_eval.evaluate_ucdd(
            file_path='test_datasets/no_drift_2d.arff',
            scaling="minmax",
            encoding="none",
            test_size=0.5,
            num_ref_batches=1,
            num_test_batches=1,
            random_state=random_state,
            additional_check=False
        )

        self.assertEqual([], drift_occurrences)

    def test_drift_three_testing_batches(self):
        drift_occurrences = ucdd_eval.evaluate_ucdd(
            file_path='test_datasets/drift_from_p21_2d.arff',
            scaling="minmax",
            encoding="none",
            test_size=0.75,
            num_ref_batches=1,
            num_test_batches=3,
            random_state=random_state,
            additional_check=False
        )

        self.assertEqual([1, 2], drift_occurrences)

    def test_drift_three_testing_batches_other_class(self):
        drift_occurrences = ucdd_eval.evaluate_ucdd(
            file_path='test_datasets/drift_from_p21_2d_other_class.arff',
            scaling="minmax",
            encoding="none",
            test_size=0.75,
            num_ref_batches=1,
            num_test_batches=3,
            random_state=random_state,
            additional_check=False
        )

        self.assertEqual([1, 2], drift_occurrences)

    def test_drift_three_testing_batches_more_neighbours_no_check(self):
        drift_occurrences = ucdd_eval.evaluate_ucdd(
            file_path='test_datasets/drift_from_p21_2d_more_neighbours.arff',
            scaling="minmax",
            encoding="none",
            test_size=0.75,
            num_ref_batches=1,
            num_test_batches=3,
            random_state=random_state,
            additional_check=False
        )

        self.assertEqual([], drift_occurrences)

    def test_drift_three_testing_batches_more_neighbours_with_check(self):
        drift_occurrences = ucdd_eval.evaluate_ucdd(
            file_path='test_datasets/drift_from_p21_2d_more_neighbours.arff',
            scaling="minmax",
            encoding="none",
            test_size=0.75,
            num_ref_batches=1,
            num_test_batches=3,
            random_state=random_state,
            additional_check=True
        )

        self.assertEqual([1, 2], drift_occurrences)


if __name__ == '__main__':
    unittest.main()
