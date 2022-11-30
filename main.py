# This is a sample Python script.
from sklearn.preprocessing import MinMaxScaler

import preprocessing
import ucdd


dataset_paths = {
    'drift_2d': 'datasets/drift_2d.arff',
    'sea_abrupt': 'datasets/sea_1_abrupt_drift_0_noise_balanced.arff',
    'agraw1_abrupt': 'datasets/agraw1_1_abrupt_drift_0_noise_balanced.arff',
    'agraw2_abrupt': 'datasets/agraw2_1_abrupt_drift_0_noise_balanced.arff'
}


if __name__ == '__main__':
    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches =\
        preprocessing.get_batches(dataset_paths['sea_abrupt'], test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                  scaling=True, scaler=MinMaxScaler())
    # preprocessing.print_batches([X_ref_batches, y_ref_batches, X_test_batches, y_test_batches],
    #                             ['reference data', 'reference labels', 'testing data', 'testing labels'])
    print('drift detected in testing batches:', ucdd.drift_occurrences_list(X_ref_batches, X_test_batches))
