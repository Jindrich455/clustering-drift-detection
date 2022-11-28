# This is a sample Python script.
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.io import arff
import pandas as pd
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

dataset_paths = {'sea_abrupt': 'datasets/sea_1_abrupt_drift_0_noise_balanced.arff',
                 'agraw1_abrupt': 'datasets/agraw1_1_abrupt_drift_0_noise_balanced.arff',
                 'agraw2_abrupt': 'datasets/agraw2_1_abrupt_drift_0_noise_balanced.arff'}


def accept_data(file_path):
    """Accept an arff file and return its contents in a pandas dataframe"""
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df


def get_pandas_reference_testing(file_path, fraction_of_ref):
    """Convert an arff file to reference and testing pandas dataframes"""
    df = accept_data(file_path)
    num_ref_rows = int(df.shape[0] * fraction_of_ref)
    num_test_rows = df.shape[0] - num_ref_rows
    print('num ref rows', num_ref_rows)
    print('num test rows', num_test_rows)
    reference = df.head(num_ref_rows)
    testing = df.tail(num_test_rows)
    print('reference #', reference.shape[0])
    print('test #', testing.shape[0])
    return reference, testing


def divide_to_batches(df_reference, num_ref_batches, df_test, num_test_batches):
    reference_batch_list = np.array_split(df_reference, num_ref_batches)
    test_batch_list = np.array_split(df_test, num_test_batches)
    return reference_batch_list, test_batch_list


def detect_cd(ref_window, test_window):
    """Detect whether a concept drift occurred based on a reference and a testing window"""


def drift_occurrences_list(reference_batch_list, test_batch_list):
    """Give a list of all batches where the algorithm detected drift"""


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def kneighbors_test():
    samples = np.array([[0., 0.], [1., 0.], [2., -1.]])
    x = np.array([[0., 1.], [1., 1.]])
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples)
    print(neigh.kneighbors([[0., 1.], [1., 1.], [2., 1.]]))


def print_batch_info(batch_list, msg):
    print('Number of ' + msg, len(batch_list))
    print('# rows per batch')
    for batch in batch_list:
        print(batch.shape[0])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_reference, df_test = get_pandas_reference_testing(dataset_paths['sea_abrupt'], 0.3)
    ref_batch_list, test_batch_list = divide_to_batches(df_reference, 3, df_test, 7)
    print_batch_info(ref_batch_list, 'reference batches')
    print_batch_info(test_batch_list, 'testing batches')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
