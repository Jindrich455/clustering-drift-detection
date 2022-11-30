# This is a sample Python script.
from sklearn.preprocessing import MinMaxScaler

import accepting
import preprocessing
import preprocessing_old
import ucdd

if __name__ == '__main__':
    df_X_ref, df_X_test, df_y_ref, df_y_test = accepting.get_pandas_reference_testing(
        'datasets/sea_1_abrupt_drift_0_noise_balanced.arff', test_fraction=0.7
    )

    df_X_ref, df_X_test = preprocessing.prepare_data(
        df_X_ref, df_X_test, scaling=True, scaler=MinMaxScaler(), use_categorical=False
    )

    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.get_batches(
        df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches=3, num_test_batches=7
    )

    drift_occurrences = ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state=0)

    # X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing_old.get_batches(
    #     'datasets/agraw1_1_abrupt_drift_0_noise_balanced.arff', test_fraction=0.7,
    #     num_ref_batches=3, num_test_batches=7, scaling=True, scaler=MinMaxScaler(),
    #     use_categorical=False)
    # # X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.get_batches(
    # #     'tests/test_datasets/drift_from_p21_2d_more_neighbours.arff', test_fraction=0.75,
    # #     num_ref_batches=1, num_test_batches=3, scaling=True, scaler=MinMaxScaler(), debug=True)
    # # preprocessing.print_batches([X_ref_batches, y_ref_batches, X_test_batches, y_test_batches],
    # #                             ['reference data', 'reference labels', 'testing data', 'testing labels'])
    print('drift detected in testing batches:', drift_occurrences)
