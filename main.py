from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import accepting
import preprocessing
import ucdd

if __name__ == '__main__':
    df_x, df_y = accepting.get_clean_df('datasets/agraw2_1_abrupt_drift_0_noise_balanced.arff')

    transformer = ColumnTransformer([
        ('num', MinMaxScaler(), selector(dtype_include='number')),
        ('cat', OneHotEncoder(sparse=False), selector(dtype_exclude='number'))
    ])
    # transformer = FunctionTransformer(lambda x: x)
    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    # ])

    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.transform_data_and_get_batches(
        df_x, df_y, test_fraction=0.7, num_ref_batches=3, num_test_batches=7, transformer=transformer
    )

    drift_occurrences = ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state=2)

    print('drift detected in testing batches:', drift_occurrences)

    # X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = preprocessing.prepare_data_and_get_batches(
    #     df_x, df_y, test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
    #     scaling=True, scaler=MinMaxScaler(), use_categorical=False, encoding=True, encoder=OneHotEncoder(sparse=False)
    # )
    #
    # drift_occurrences = ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state=2)
    #
    # print('drift detected in testing batches:', drift_occurrences)
