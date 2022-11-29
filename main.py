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
    df_X_ref, df_X_test, df_y_ref, df_y_test =\
        preprocessing.get_pandas_reference_testing(dataset_paths['drift_2d'], 0.5)
    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches =\
        preprocessing.divide_to_batches(df_X_ref, df_y_ref, 1, df_X_test, df_y_test, 1)
    preprocessing.print_batches([X_ref_batches, y_ref_batches, X_test_batches, y_test_batches],
                                ['reference data', 'reference labels', 'testing data', 'testing labels'])
    drift_happening = ucdd.detect_cd(df_X_ref, df_X_test)
    if drift_happening:
        print('DRIFT!!!!')
