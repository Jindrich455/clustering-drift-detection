import os
import time

import numpy as np
import pandas as pd
import pyclustering.utils
import scipy.spatial.distance
import sklearn.preprocessing
from category_encoders import TargetEncoder
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import type_metric, distance_metric
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder

import accepting
import mssw.mssw
import mssw.mssw_eval
import mssw.mssw_eval_local_datasets
from ucdd import ucdd_supported_parameters as ucdd_spms, ucdd_eval_and_write_res, ucdd_eval, ucdd_read_and_evaluate
import mssw.mssw_supported_parameters as mssw_spms


def big_evaluation0():
    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=['Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff'],
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True],
        detect_all_training_batches_list=[False],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN],
        use_pyclustering=True
    )


def big_evaluation():
    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=['Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff'],
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True, False],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN, ucdd_spms.Distances.MANHATTAN],
        use_pyclustering=True
    )


def big_evaluation2():
    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=[
            'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff',
            'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff'
        ],
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE, ucdd_spms.Encoders.ONEHOT, ucdd_spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True, False],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN, ucdd_spms.Distances.MANHATTAN],
        use_pyclustering=True
    )


def big_evaluation3():
    paths = [
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_20.arff'
    ]

    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=paths,
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[False],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN, ucdd_spms.Distances.MANHATTAN],
        use_pyclustering=True
    )


def big_evaluation4():
    paths = [
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_20.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_20.arff'
    ]

    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=paths,
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE, ucdd_spms.Encoders.ONEHOT, ucdd_spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[False],
        detect_all_training_batches_list=[False],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN],
        use_pyclustering=True
    )


def big_evaluation5():
    paths = [
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_20.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_20.arff'
    ]

    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=paths,
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE, ucdd_spms.Encoders.ONEHOT, ucdd_spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True],
        detect_all_training_batches_list=[False],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN],
        use_pyclustering=True
    )


def big_evaluation6():
    paths = [
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_20.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_20.arff'
    ]

    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=paths,
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE, ucdd_spms.Encoders.ONEHOT, ucdd_spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[False],
        metric_ids=[ucdd_spms.Distances.MANHATTAN],
        use_pyclustering=True
    )


def big_evaluation7():
    paths = [
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_20.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_20.arff'
    ]

    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=paths,
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE, ucdd_spms.Encoders.ONEHOT, ucdd_spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN, ucdd_spms.Distances.MANHATTAN],
        use_pyclustering=True
    )


def big_evaluation8():
    paths = [
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_1.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_5.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_05.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_10.arff',
        'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_20.arff'
    ]

    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=paths,
        scalings=[ucdd_spms.Scalers.MINMAX],
        encodings=[ucdd_spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True],
        metric_ids=[ucdd_spms.Distances.EUCLIDEAN, ucdd_spms.Distances.MANHATTAN],
        use_pyclustering=True
    )


def save_metrics():
    path = 'ucdd/runs_results/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = ucdd_read_and_evaluate.all_info_for_all_files_in_folder(path)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = ucdd_read_and_evaluate.rows_with_drift_detected(all_info_df)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_metrics2():
    path = 'ucdd/runs_results/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = ucdd_read_and_evaluate.all_info_for_all_files_in_folder(path)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = ucdd_read_and_evaluate.rows_with_drift_detected(all_info_df)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_metrics3():
    path = 'ucdd/runs_results/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = ucdd_read_and_evaluate.all_info_for_all_files_in_folder(path)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = ucdd_read_and_evaluate.rows_with_drift_detected(all_info_df)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_metrics4():
    path = 'ucdd/runs_results/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_1'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = ucdd_read_and_evaluate.all_info_for_all_files_in_folder(path)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = ucdd_read_and_evaluate.rows_with_drift_detected(all_info_df)
    ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_all_metrics():
    dir_path = 'ucdd/runs_results/synthetic_data/gradual_drift'
    directory_names = os.listdir(dir_path)
    for name in directory_names:
        path = dir_path + '/' + name
        all_info_df = ucdd_read_and_evaluate.all_info_for_all_files_in_folder(path)
        ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
        useful_info_df = ucdd_read_and_evaluate.rows_with_drift_detected(all_info_df)
        ucdd_read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def perform_clustering():
    dataset_path = 'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff'
    x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = ucdd_eval.obtain_preprocessed_batches(
        dataset_path,
        ucdd_spms.Scalers.MINMAX,
        ucdd_spms.Encoders.ONEHOT,
        0.5,
        1,
        1
    )
    print(x_test_batches)
    # x_test_batches = [x_test_batches[0].to_numpy()]

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    kmeans_instance = kmeans(x_test_batches[0], initial_centers)
    kmeans_instance.process()

    print('Execution time of the default pyclustering k-means:', time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED,
                                func=pyclustering.utils.euclidean_distance_square)
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with injected pyclustering euclidean square:', time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED,
                                func=pyclustering.utils.euclidean_distance_square)
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with injected pyclustering euclidean:',
          time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED,
                                func=lambda o1, o2: np.sqrt(np.sum(np.square(np.subtract(o1, o2)))))
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with injected numpy euclidean using subtract:',
          time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED,
                                func=lambda o1, o2: np.sqrt(np.sum(np.square(o1 - o2))))
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with injected numpy euclidean:',
          time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED,
                                func=euclidean_distance)
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with injected simple euclidean:',
          time.time() - start)

    # start = time.time()
    #
    # w = [0.3, 0.5, 0.2]
    # initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    # my_metric = distance_metric(type_metric.USER_DEFINED,
    #                             func=lambda u, v: euclidean_distance_weighted(u, v, w))
    # kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    # kmeans_instance.process()
    #
    # print('Execution time of the pyclustering k-means with injected weighted euclidean:',
    #       time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED,
                                func=lambda u, v: euclidean_distance_numpy(u, v))
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with injected euclidean in a function using numpy:',
          time.time() - start)

    # start = time.time()
    #
    # w = np.array([0.3, 0.5, 0.2])
    # initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    # my_metric = distance_metric(type_metric.USER_DEFINED,
    #                             func=lambda u, v: euclidean_distance_weighted_numpy(u, v, w))
    # kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    # kmeans_instance.process()
    #
    # print('Execution time of the pyclustering k-means with injected weighted euclidean in a function using numpy:',
    #       time.time() - start)

    # start = time.time()
    #
    # w = np.array([0.3, 0.5, 0.2])
    # initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    # my_metric = distance_metric(type_metric.USER_DEFINED,
    #                             func=lambda u, v: euclidean_distance_weighted_numpy(u, v, w))
    # kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    # kmeans_instance.process()
    #
    # print('Execution time of the pyclustering k-means with injected weighted euclidean in a function using numpy:',
    #       time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED, func=lambda u, v: scipy.spatial.distance.euclidean(u, v, [0.3, 0.5, 0.2]))
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with scipy weighted euclidean:', time.time() - start)

    start = time.time()

    initial_centers = kmeans_plusplus_initializer(x_test_batches[0], 2).initialize()
    my_metric = distance_metric(type_metric.USER_DEFINED,
                                func=scipy.spatial.distance.euclidean)
    kmeans_instance = kmeans(x_test_batches[0], initial_centers, metric=my_metric)
    kmeans_instance.process()

    print('Execution time of the pyclustering k-means with scipy default euclidean:', time.time() - start)


def euclidean_distance(o1, o2):
    distance = 0.0
    for i in range(len(o1)):
        distance += (o1[i] - o2[i]) ** 2.0
    return distance ** 0.5


def euclidean_distance_weighted(o1, o2, w):
    distance = 0.0
    for i in range(len(o1)):
        distance += w[i] * ((o1[i] - o2[i]) ** 2.0)
    return distance ** 0.5


def euclidean_distance_weighted_numpy(o1, o2, w):
    return np.sqrt(np.sum(np.multiply(w, np.square(np.subtract(o1, o2)))))


def euclidean_distance_numpy(o1, o2):
    return np.sqrt(np.sum(np.square(np.subtract(o1, o2))))


def mssw_eval_attempt():
    data_path = 'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_05.arff'
    df_x, _ = accepting.get_clean_df(data_path)
    numpy_data = df_x.to_numpy()

    reference_data = numpy_data[:30000]
    testing_data = numpy_data[30000:]
    print('ref')
    print(reference_data)
    print('test')
    print(testing_data)
    ref_batches = np.array_split(reference_data, 3)
    test_batches = np.array_split(testing_data, 7)
    print('num ref batches')
    print(len(ref_batches))
    print(ref_batches)
    print('num test batches')
    print(len(test_batches))
    print(test_batches)

    drifts_detected = mssw.mssw.all_drifting_batches(ref_batches, test_batches, num_clusters=2)
    print('drifts detected')
    print(drifts_detected)


def mssw_eval_attempt2():
    data_path = 'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff'
    df_x, df_y = accepting.get_clean_df(data_path)

    df_y = pd.DataFrame(LabelEncoder().fit_transform(df_y))
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=0.7, shuffle=False)

    df_x_ref_num, df_x_ref_cat = accepting.divide_numeric_categorical(df_x_ref)
    df_x_test_num, df_x_test_cat = accepting.divide_numeric_categorical(df_x_test)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(df_x_ref_cat)

    ref_index = df_x_ref_cat.index
    test_index = df_x_test_cat.index
    df_x_ref_cat_transformed = pd.DataFrame(encoder.transform(df_x_ref_cat))
    df_x_test_cat_transformed = pd.DataFrame(encoder.transform(df_x_test_cat))
    df_x_ref_cat_transformed.set_index(ref_index, inplace=True)
    df_x_test_cat_transformed.set_index(test_index, inplace=True)

    # print('df_x_ref_num')
    # print(df_x_ref_num)
    # print('df_x_ref_cat_transformed')
    # print(df_x_ref_cat_transformed)
    # print('df_x_test_num')
    # print(df_x_test_num)
    # print('df_x_test_cat_transformed')
    # print(df_x_test_cat_transformed)

    reference_data = df_x_ref_num.join(df_x_ref_cat_transformed, lsuffix='_num').to_numpy()
    testing_data = df_x_test_num.join(df_x_test_cat_transformed, lsuffix='_num').to_numpy()

    ref_batches = np.array_split(reference_data, 3)
    test_batches = np.array_split(testing_data, 7)
    print('num ref batches')
    print(len(ref_batches))
    print(ref_batches)
    print('num test batches')
    print(len(test_batches))
    print(test_batches)

    start = time.time()
    drifts_detected = mssw.mssw.all_drifting_batches(ref_batches, test_batches, num_clusters=2)
    print('drifts detected')
    print(drifts_detected)

    print('time taken')
    print(time.time() - start)


def mssw_eval_attempt3():
    data_path = 'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_5.arff'
    df_x, df_y = accepting.get_clean_df(data_path)

    df_y = pd.DataFrame(LabelEncoder().fit_transform(df_y))
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=0.7, shuffle=False)

    df_x_ref_num, df_x_ref_cat = accepting.divide_numeric_categorical(df_x_ref)
    df_x_test_num, df_x_test_cat = accepting.divide_numeric_categorical(df_x_test)

    encoder = TargetEncoder()
    encoder.fit(df_x_ref_cat, df_y_ref)

    ref_index = df_x_ref_cat.index
    test_index = df_x_test_cat.index
    df_x_ref_cat_transformed = pd.DataFrame(encoder.transform(df_x_ref_cat))
    df_x_test_cat_transformed = pd.DataFrame(encoder.transform(df_x_test_cat))
    df_x_ref_cat_transformed.set_index(ref_index, inplace=True)
    df_x_test_cat_transformed.set_index(test_index, inplace=True)

    # print('df_x_ref_num')
    # print(df_x_ref_num)
    # print('df_x_ref_cat_transformed')
    # print(df_x_ref_cat_transformed)
    # print('df_x_test_num')
    # print(df_x_test_num)
    # print('df_x_test_cat_transformed')
    # print(df_x_test_cat_transformed)

    reference_data = df_x_ref_num.join(df_x_ref_cat_transformed, lsuffix='_num').to_numpy()
    testing_data = df_x_test_num.join(df_x_test_cat_transformed, lsuffix='_num').to_numpy()

    ref_batches = np.array_split(reference_data, 3)
    test_batches = np.array_split(testing_data, 7)
    print('num ref batches')
    print(len(ref_batches))
    print(ref_batches)
    print('num test batches')
    print(len(test_batches))
    print(test_batches)

    start = time.time()
    drifts_detected = mssw.mssw.all_drifting_batches(ref_batches, test_batches, num_clusters=2)
    print('drifts detected')
    print(drifts_detected)

    print('time taken')
    print(time.time() - start)


def mssw_eval_attempt4():
    data_path = 'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_20.arff'
    df_x, _ = accepting.get_clean_df(data_path)
    numpy_data = df_x.to_numpy()

    reference_data = numpy_data[:30000]
    testing_data = numpy_data[30000:]
    print('ref')
    print(reference_data)
    print('test')
    print(testing_data)
    ref_batches = np.array_split(reference_data, 3)
    test_batches = np.array_split(testing_data, 7)
    print('num ref batches')
    print(len(ref_batches))
    print(ref_batches)
    print('num test batches')
    print(len(test_batches))
    print(test_batches)

    runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se = mssw.mssw_eval\
        .all_drifting_batches_randomness_robust(ref_batches, test_batches)


def mssw_eval_attempt5():
    data_path = 'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_10.arff'
    df_x, df_y = accepting.get_clean_df(data_path)

    df_y = pd.DataFrame(LabelEncoder().fit_transform(df_y))
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=0.7, shuffle=False)

    df_x_ref_num, df_x_ref_cat = accepting.divide_numeric_categorical(df_x_ref)
    df_x_test_num, df_x_test_cat = accepting.divide_numeric_categorical(df_x_test)

    encoder = TargetEncoder()
    encoder.fit(df_x_ref_cat, df_y_ref)

    ref_index = df_x_ref_cat.index
    test_index = df_x_test_cat.index
    df_x_ref_cat_transformed = pd.DataFrame(encoder.transform(df_x_ref_cat))
    df_x_test_cat_transformed = pd.DataFrame(encoder.transform(df_x_test_cat))
    df_x_ref_cat_transformed.set_index(ref_index, inplace=True)
    df_x_test_cat_transformed.set_index(test_index, inplace=True)

    # print('df_x_ref_num')
    # print(df_x_ref_num)
    # print('df_x_ref_cat_transformed')
    # print(df_x_ref_cat_transformed)
    # print('df_x_test_num')
    # print(df_x_test_num)
    # print('df_x_test_cat_transformed')
    # print(df_x_test_cat_transformed)

    reference_data = df_x_ref_num.join(df_x_ref_cat_transformed, lsuffix='_num').to_numpy()
    testing_data = df_x_test_num.join(df_x_test_cat_transformed, lsuffix='_num').to_numpy()

    ref_batches = np.array_split(reference_data, 3)
    test_batches = np.array_split(testing_data, 7)
    print('num ref batches')
    print(len(ref_batches))
    print(ref_batches)
    print('num test batches')
    print(len(test_batches))
    print(test_batches)

    start = time.time()
    runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se = mssw.mssw_eval \
        .all_drifting_batches_randomness_robust(ref_batches, test_batches)

    print('time taken')
    print(time.time() - start)


def mssw_eval_attempt6():
    start = time.time()
    runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se =\
        mssw.mssw_eval_local_datasets.eval_one_parameter_set(
            'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff',
            mssw_spms.Encoders.ONEHOT,
            test_fraction=0.7,
            num_ref_batches=3,
            num_test_batches=7,
            true_drift_idx=2
        )

    print('mean FPR', fpr_mean)
    print('mean latency', latency_mean)

    print('time taken')
    print(time.time() - start)


def mssw_big_eval():
    results = mssw.mssw_eval_local_datasets.eval_multiple_parameter_sets(
        ['Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff',
         'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff'],
        [mssw_spms.Encoders.ONEHOT, mssw_spms.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2
    )
    print(results)
