# This is a sample Python script.
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.io import arff
import pandas as pd
import numpy as np

dataset_paths = {
    '2d_testing': 'datasets/test_2d.arff',
    'sea_abrupt': 'datasets/sea_1_abrupt_drift_0_noise_balanced.arff',
    'agraw1_abrupt': 'datasets/agraw1_1_abrupt_drift_0_noise_balanced.arff',
    'agraw2_abrupt': 'datasets/agraw2_1_abrupt_drift_0_noise_balanced.arff'
}


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


def join_predict_split(df_ref_window, df_test_window):
    """Join points from two windows, predict their labels through kmeans, then separate them again"""
    df_ref_window.insert(df_ref_window.shape[1], 'window', 'ref')
    df_test_window.insert(df_test_window.shape[1], 'window', 'test')
    print(df_ref_window)
    df_union = pd.concat([df_ref_window, df_test_window])
    print(df_union)
    df_class_window = df_union[['class', 'window']]
    df_no_labels = df_union.drop(columns=['class', 'window'])
    print(df_no_labels.head())
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df_no_labels)
    labels = KMeans(n_clusters=2, random_state=0).fit_predict(df_no_labels)
    print(kmeans.cluster_centers_)
    print('labels', labels)
    df_predicted_labels = df_no_labels
    df_predicted_labels['labelprediction'] = labels
    df_predicted_labels = df_predicted_labels.join(df_class_window)
    print('df_predicted_labels\n', df_predicted_labels)

    ref_mask = df_predicted_labels['window'] == 'ref'
    plus_mask = df_predicted_labels['labelprediction'] == 1

    ref_plus_mask = ref_mask & plus_mask
    ref_minus_mask = ref_mask & (~plus_mask)
    test_plus_mask = (~ref_mask) & plus_mask
    test_minus_mask = (~ref_mask) & (~plus_mask)

    df_ref_plus = df_predicted_labels[ref_plus_mask]
    df_ref_minus = df_predicted_labels[ref_minus_mask]
    df_test_plus = df_predicted_labels[test_plus_mask]
    df_test_minus = df_predicted_labels[test_minus_mask]

    print('df_ref_plus', df_ref_plus)
    print('df_test_minus', df_test_minus)

    return df_ref_plus, df_ref_minus, df_test_plus, df_test_minus


def get_u_v0_v1(df_u, df_v0, df_v1):
    columns_to_drop = ['labelprediction', 'class', 'window']
    u = df_u.drop(columns=columns_to_drop)
    v0 = df_v0.drop(columns=columns_to_drop)
    v1 = df_v1.drop(columns=columns_to_drop)

    return u, v0, v1


def compute_beta(df_u, df_v0, df_v1):
    u, v0, v1 = get_u_v0_v1(df_u, df_v0, df_v1)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(u)
    neighbors_v0 = neigh.kneighbors(v0)
    print('neighbors_v0', neighbors_v0)


def detect_cd(df_ref_window, df_test_window):
    """Detect whether a concept drift occurred based on a reference and a testing window"""
    df_ref_plus, df_ref_minus, df_test_plus, df_test_minus = join_predict_split(df_ref_window, df_test_window)
    beta_minus = compute_beta(df_ref_plus, df_ref_minus, df_test_minus)
    beta_plus = compute_beta(df_ref_minus, df_ref_plus, df_test_plus)

def drift_occurrences_list(reference_batch_list, test_batch_list):
    """Return a list of all batches where the algorithm detected drift"""


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
    df_reference, df_test = get_pandas_reference_testing(dataset_paths['2d_testing'], 0.5)
    ref_batch_list, test_batch_list = divide_to_batches(df_reference, 1, df_test, 1)
    print_batch_info(ref_batch_list, 'reference batches')
    print_batch_info(test_batch_list, 'testing batches')
    detect_cd(ref_batch_list[0], test_batch_list[0])
