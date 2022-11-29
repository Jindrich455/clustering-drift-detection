import sklearn.model_selection
from scipy.io import arff
import pandas as pd
import numpy as np

# LABELENCODER!!!
# LABELENCODER!!!
# LABELENCODER!!!


def accept_data(file_path):
    """Accept an arff file and return its contents in a pandas dataframe"""
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df


def get_pandas_reference_testing(file_path, test_fraction=0.7):
    """Convert an arff file to reference and testing pandas dataframes"""
    df = accept_data(file_path)
    print('accepted df\n', df)
    df_x = df.drop(columns='class')
    df_y = df[['class']]
    reference, testing, ref_labels, test_labels =\
        sklearn.model_selection.train_test_split(df_x, df_y, test_size=test_fraction, shuffle=False)
    print('reference', reference)
    print('ref_labels', ref_labels)
    print('testing', testing)
    print('test_labels', test_labels)
    return reference, testing


def divide_to_batches(df_reference, num_ref_batches, df_test, num_test_batches):
    reference_batch_list = np.array_split(df_reference, num_ref_batches)
    test_batch_list = np.array_split(df_test, num_test_batches)
    return reference_batch_list, test_batch_list


def print_batch_info(batch_list, msg):
    print('Number of ' + msg, len(batch_list))
    print('# rows per batch')
    for batch in batch_list:
        print(batch.shape[0])
