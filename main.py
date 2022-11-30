# This is a sample Python script.
from sklearn.preprocessing import MinMaxScaler

import preprocessing
import ucdd

if __name__ == '__main__':
    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches =preprocessing.get_batches(
        'datasets/sea_1_abrupt_drift_0_noise_balanced.arff', test_fraction=0.7,
        num_ref_batches=3, num_test_batches=7, scaling=True, scaler=MinMaxScaler())
    # X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.get_batches(
    #     'tests/test_datasets/drift_from_p21_2d_more_neighbours.arff', test_fraction=0.75,
    #     num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler(), debug=True)
    # preprocessing.print_batches([X_ref_batches, y_ref_batches, X_test_batches, y_test_batches],
    #                             ['reference data', 'reference labels', 'testing data', 'testing labels'])
    print('drift detected in testing batches:',
          ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state=0))
