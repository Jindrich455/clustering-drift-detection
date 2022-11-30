import sklearn.model_selection
from scipy.io import arff
import pandas as pd
import numpy as np

# LABELENCODER!!!
# LABELENCODER!!!
# LABELENCODER!!!
from sklearn.preprocessing import MinMaxScaler


def accept_data(file_path):
    """Accept an arff file and return its contents in a pandas dataframe"""
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df


def get_pandas_reference_testing(file_path, test_fraction, scaling, scaler):
    """Convert an arff file to reference and testing pandas dataframes"""
    df = accept_data(file_path)
    df_x = df.drop(columns='class')
    if scaling:
        df_x[df_x.columns] = scaler.fit_transform(df_x[df_x.columns])
    df_y = df[['class']]
    df_X_ref, df_X_test, df_y_ref, df_y_test =\
        sklearn.model_selection.train_test_split(df_x, df_y, test_size=test_fraction, shuffle=False)
    return df_X_ref, df_X_test, df_y_ref, df_y_test


def divide_to_batches(df_X_ref, df_y_ref, num_ref_batches, df_X_test, df_y_test, num_test_batches):
    X_ref_batches = np.array_split(df_X_ref, num_ref_batches)
    y_ref_batches = np.array_split(df_y_ref, num_ref_batches)
    X_test_batches = np.array_split(df_X_test, num_test_batches)
    y_test_batches = np.array_split(df_y_test, num_test_batches)
    return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches


def get_batches(file_path, test_fraction, num_ref_batches, num_test_batches, scaling, scaler, debug=False):
    df_X_ref, df_X_test, df_y_ref, df_y_test = \
        get_pandas_reference_testing(file_path, test_fraction, scaling, scaler)
    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches =\
        divide_to_batches(df_X_ref, df_y_ref, num_ref_batches, df_X_test, df_y_test, num_test_batches)
    if debug:
        print_batches([X_ref_batches, y_ref_batches, X_test_batches, y_test_batches],
                     ['reference data', 'reference labels', 'testing data', 'testing labels'])
    return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches


def print_batch_info(batch_list, msg):
    print('Number of batches with ' + msg + ':', len(batch_list))
    print('# rows per batch')
    for batch in batch_list:
        print(batch.shape[0])
        print('First 10 entries:')
        print(batch.head())


def print_batches(batches, batch_msgs):
    zipped = zip(batches, batch_msgs)
    for batch, msg in zipped:
        print_batch_info(batch, msg)

