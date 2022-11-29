# This is a sample Python script.
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
        preprocessing.get_batches(dataset_paths['drift_2d'], test_fraction=0.5, num_ref_batches=1, num_test_batches=1)
    preprocessing.print_batches([X_ref_batches, y_ref_batches, X_test_batches, y_test_batches],
                                ['reference data', 'reference labels', 'testing data', 'testing labels'])
    drift_happening = ucdd.detect_cd(X_ref_batches[0], X_test_batches[0])
    if drift_happening:
        print('DRIFT!!!!')
