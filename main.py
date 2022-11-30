# This is a sample Python script.
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import accepting
import preprocessing
import ucdd

if __name__ == '__main__':
    df_x, df_y = accepting.get_clean_df('datasets/agraw1_1_abrupt_drift_0_noise_balanced.arff')
    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.prepare_data_and_get_batches(
        df_x, df_y, test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
        scaling=True, scaler=MinMaxScaler(), use_categorical=False, encoding=False, encoder=OneHotEncoder(sparse=False)
    )

    drift_occurrences = ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state=0)

    print('drift detected in testing batches:', drift_occurrences)
