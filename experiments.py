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
import mssw.mssw_result_writer
import ucdd_improved.ucdd
import ucdd_improved.ucdd_eval
import ucdd_improved.ucdd_eval_local_datasets
import ucdd_improved.ucdd_result_writer
import ucdd_improved.ucdd_supported_parameters
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

    runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se = mssw.mssw_eval \
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
    runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se = \
        mssw.mssw_eval_local_datasets.eval_one_parameter_set(
            'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff',
            mssw_spms.Encoders.ONEHOT, test_fraction=0.7, num_ref_batches=3, num_test_batches=7, true_drift_idx=2)

    print('mean FPR', fpr_mean)
    print('mean latency', latency_mean)

    print('time taken')
    print(time.time() - start)


def mssw_big_eval():
    results = mssw.mssw_eval_local_datasets.eval_multiple_parameter_sets(
        ['Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff',
         'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff'],
        [mssw_spms.Encoders.ONEHOT, mssw_spms.Encoders.TARGET], test_fraction=0.7, num_ref_batches=3,
        num_test_batches=7, true_drift_idx=2)
    print(results)


abrupt_sea_path = 'Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff'
abrupt_agraw1_path = 'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff'
abrupt_agraw2_path = 'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff'

gradual_sea_paths = [
    'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_05.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_1.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_5.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_10.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_20.arff'
]

gradual_agraw1_paths = [
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_05.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_1.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_5.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_10.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_20.arff'
]

gradual_agraw2_paths = [
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_05.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_1.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_5.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_10.arff',
    'Datasets_concept_drift/synthetic_data/gradual_drift/agraw2_1_gradual_drift_0_noise_balanced_20.arff'
]

only_numerical_data_paths = [abrupt_sea_path] + gradual_sea_paths
only_mixed_data_paths = [abrupt_agraw1_path] + gradual_agraw1_paths + [abrupt_agraw2_path] + gradual_agraw2_paths


def mssw_big_eval_write_res():
    results = mssw.mssw_result_writer.eval_and_write_raw(
        ['Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff',
         'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff'],
        [mssw_spms.Encoders.ONEHOT, mssw_spms.Encoders.TARGET], test_fraction=0.7, num_ref_batches=3,
        num_test_batches=7, true_drift_idx=2)
    print(results)


def mssw_big_eval_write_res2():
    mssw.mssw_result_writer.eval_and_write_raw(only_numerical_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                               test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                               true_drift_idx=2)
    mssw.mssw_result_writer.eval_and_write_raw(only_mixed_data_paths,
                                               [mssw_spms.Encoders.EXCLUDE, mssw_spms.Encoders.ONEHOT,
                                                mssw_spms.Encoders.TARGET], test_fraction=0.7, num_ref_batches=3,
                                               num_test_batches=7, true_drift_idx=2)


def mssw_combine_results():
    mssw.mssw_result_writer.combine_synthetic_results()


def mssw_write_to_file():
    mssw.mssw_result_writer.eval_and_write_to_file(only_numerical_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100],
                                                   max_iters=[280],
                                                   tols=[2e-4],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_test_different_clustering_params.csv')


def mssw_write_to_file_mixed_exclude():
    mssw.mssw_result_writer.eval_and_write_to_file(only_mixed_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100],
                                                   max_iters=[1000],
                                                   tols=[2e-4],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_mixed_exclude_high_max_iter.csv')


def mssw_write_to_file_multiple_clustering_possibilities():
    mssw.mssw_result_writer.eval_and_write_to_file(only_mixed_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100, 200],
                                                   max_iters=[100, 1000],
                                                   tols=[1e-2, 2e-4],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_mixed_exclude_multiple_clustering_params.csv')


def mssw_write_to_file_multiple_encoding_strategies():
    mssw.mssw_result_writer.eval_and_write_to_file(only_mixed_data_paths, [mssw_spms.Encoders.EXCLUDE, mssw_spms.Encoders.ONEHOT],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100],
                                                   max_iters=[1000],
                                                   tols=[1e-2, 1e-4],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_mixed_exclude_multiple_encoding.csv')


def mssw_write_to_file_progressively_changing_clustering_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file(only_numerical_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100],
                                                   max_iters=[1000],
                                                   tols=[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_numerical_decreasing_tol.csv')

    mssw.mssw_result_writer.eval_and_write_to_file(only_mixed_data_paths, [mssw_spms.Encoders.EXCLUDE,
                                                                           mssw_spms.Encoders.ONEHOT,
                                                                           mssw_spms.Encoders.TARGET],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100],
                                                   max_iters=[1000],
                                                   tols=[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_mixed_decreasing_tol.csv')

    mssw.mssw_result_writer.eval_and_write_to_file(only_numerical_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=10,
                                                   n_inits=[1, 10, 20, 50, 100, 200],
                                                   max_iters=[1000],
                                                   tols=[0.0001],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_numerical_increasing_n_init.csv')

    mssw.mssw_result_writer.eval_and_write_to_file(only_mixed_data_paths, [mssw_spms.Encoders.EXCLUDE,
                                                                           mssw_spms.Encoders.ONEHOT,
                                                                           mssw_spms.Encoders.TARGET],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=10,
                                                   n_inits=[1, 10, 20, 50, 100, 200],
                                                   max_iters=[1000],
                                                   tols=[0.0001],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_mixed_increasing_n_init.csv')

    mssw.mssw_result_writer.eval_and_write_to_file(only_numerical_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100],
                                                   max_iters=[2, 5, 10, 20, 50, 100, 1000],
                                                   tols=[0.0001],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_numerical_increasing_max_iter.csv')

    mssw.mssw_result_writer.eval_and_write_to_file(only_mixed_data_paths, [mssw_spms.Encoders.EXCLUDE,
                                                                           mssw_spms.Encoders.ONEHOT,
                                                                           mssw_spms.Encoders.TARGET],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=2,
                                                   n_inits=[100],
                                                   max_iters=[2, 5, 10, 20, 50, 100, 1000],
                                                   tols=[0.0001],
                                                   n_clusters=2,
                                                   result_file='mssw/results/mssw_mixed_increasing_max_iter.csv')


def mssw_write_to_file_sea_with_best_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file(only_numerical_data_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=5,
                                                   n_inits=[100],
                                                   max_iters=[550],
                                                   tols=[0],
                                                   n_clusters=2,
                                                   result_file='mssw/results_after_analysis/sea.csv')


def mssw_write_to_file_agraw1_exclude_with_best_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file([abrupt_agraw1_path] + gradual_agraw1_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=5,
                                                   n_inits=[100],
                                                   max_iters=[250],
                                                   tols=[0],
                                                   n_clusters=2,
                                                   result_file='mssw/results_after_analysis/agraw1_exclude.csv')


def mssw_write_to_file_agraw1_onehot_with_best_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file([abrupt_agraw1_path] + gradual_agraw1_paths, [mssw_spms.Encoders.ONEHOT],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=5,
                                                   n_inits=[100],
                                                   max_iters=[200],
                                                   tols=[0],
                                                   n_clusters=2,
                                                   result_file='mssw/results_after_analysis/agraw1_onehot.csv')


def mssw_write_to_file_agraw1_target_with_best_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file([abrupt_agraw1_path] + gradual_agraw1_paths, [mssw_spms.Encoders.TARGET],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=5,
                                                   n_inits=[100],
                                                   max_iters=[300],
                                                   tols=[0],
                                                   n_clusters=2,
                                                   result_file='mssw/results_after_analysis/agraw1_target.csv')


def mssw_write_to_file_agraw2_exclude_with_best_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file([abrupt_agraw2_path] + gradual_agraw2_paths, [mssw_spms.Encoders.EXCLUDE],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=5,
                                                   n_inits=[100],
                                                   max_iters=[300],
                                                   tols=[0],
                                                   n_clusters=2,
                                                   result_file='mssw/results_after_analysis/agraw2_exclude.csv')


def mssw_write_to_file_agraw2_onehot_with_best_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file([abrupt_agraw2_path] + gradual_agraw2_paths, [mssw_spms.Encoders.ONEHOT],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=5,
                                                   n_inits=[100],
                                                   max_iters=[300],
                                                   tols=[0],
                                                   n_clusters=2,
                                                   result_file='mssw/results_after_analysis/agraw2_onehot.csv')


def mssw_write_to_file_agraw2_target_with_best_parameters():
    mssw.mssw_result_writer.eval_and_write_to_file([abrupt_agraw2_path] + gradual_agraw2_paths, [mssw_spms.Encoders.TARGET],
                                                   test_fraction=0.7, num_ref_batches=3, num_test_batches=7,
                                                   true_drift_idx=2,
                                                   min_runs=5,
                                                   n_inits=[100],
                                                   max_iters=[250],
                                                   tols=[0],
                                                   n_clusters=2,
                                                   result_file='mssw/results_after_analysis/agraw2_target.csv')


def ucdd_improved_simple():
    df_x, df_y = accepting.get_clean_df(
        'Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff')

    df_y = pd.DataFrame(LabelEncoder().fit_transform(df_y))
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=0.7, shuffle=False)

    scaler = MinMaxScaler()
    scaler.fit(df_x_ref)
    reference_data = pd.DataFrame(scaler.transform(df_x_ref)).to_numpy()
    testing_data = pd.DataFrame(scaler.transform(df_x_test)).to_numpy()

    ref_batches = np.array_split(reference_data, 3)
    test_batches = np.array_split(testing_data, 7)

    drifts = ucdd_improved.ucdd.all_drifting_batches(
        ref_batches,
        test_batches,
        train_batch_strategy=ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY,
        additional_check=True,
        random_state=0
    )
    print(drifts)


def ucdd_improved_randomness_robust():
    df_x, df_y = accepting.get_clean_df(
        'Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff')

    df_y = pd.DataFrame(LabelEncoder().fit_transform(df_y))
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=0.7, shuffle=False)

    scaler = MinMaxScaler()
    scaler.fit(df_x_ref)
    reference_data = pd.DataFrame(scaler.transform(df_x_ref)).to_numpy()
    testing_data = pd.DataFrame(scaler.transform(df_x_test)).to_numpy()

    ref_batches = np.array_split(reference_data, 3)
    test_batches = np.array_split(testing_data, 7)

    runs_results_bool, final_fpr_mean, fpr_std_err, final_latency_mean, latency_std_err = ucdd_improved.ucdd_eval.all_drifting_batches_randomness_robust(
        ref_batches,
        test_batches,
        train_batch_strategy=ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY,
        additional_check=True
    )

    print(runs_results_bool, final_fpr_mean, fpr_std_err, final_latency_mean, latency_std_err)


def ucdd_improved_automated_one_dataset():
    path = 'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff'
    runs_results_bool, final_fpr_mean, fpr_std_err, final_latency_mean, latency_std_err = ucdd_improved.ucdd_eval_local_datasets.eval_one_parameter_set(
        path,
        encoding=ucdd_improved.ucdd_supported_parameters.Encoders.TARGET,
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        train_batch_strategy=ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY,
        additional_check=True
    )

    print(runs_results_bool, final_fpr_mean, fpr_std_err, final_latency_mean, latency_std_err)


def ucdd_improved_automated_multiple_parameter_sets():
    argument_results = ucdd_improved.ucdd_eval_local_datasets.eval_multiple_parameter_sets(
        data_paths=only_numerical_data_paths[:2],
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY],
        additional_checks=[True]
    )

    print(argument_results)


def ucdd_improved_write_to_file():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths[:2],
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        result_file='ucdd_improved/results/first_result_numerical.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY],
        additional_checks=[True]
    )


def ucdd_improved_write_to_file_all_numerical():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        result_file='ucdd_improved/results/all_numerical_results.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY,
                                ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.MAJORITY,
                                ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ALL],
        additional_checks=[True, False]
    )


def ucdd_improved_write_to_file_all_numerical_small_max_iter():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        max_iters=[10],
        result_file='ucdd_improved/results/all_numerical_results_small_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY,
                                ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.MAJORITY,
                                ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ALL],
        additional_checks=[True, False]
    )


def ucdd_improved_write_to_file_trainbatchstrategy_majority():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_majority/all_numerical_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.MAJORITY],
        additional_checks=[True, False]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_majority/mixed_exclude_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.MAJORITY],
        additional_checks=[True, False]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_majority/mixed_target_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.MAJORITY],
        additional_checks=[True, False]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_majority/mixed_onehot_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.MAJORITY],
        additional_checks=[True, False]
    )


def ucdd_improved_eval_new_rand_state_change():
    argument_results = ucdd_improved.ucdd_eval_local_datasets.eval_multiple_parameter_sets(
        data_paths=only_numerical_data_paths[0:1],
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        min_runs=10,
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY],
        additional_checks=[True]
    )

    print(argument_results)


def ucdd_improved_write_to_file_trainbatchstrategy_any():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_any/all_numerical_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY],
        additional_checks=[True, False]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_any/mixed_exclude_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY],
        additional_checks=[True, False]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_any/mixed_target_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY],
        additional_checks=[True, False]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_trainbatchstrat_any/mixed_onehot_many_parameters.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.ANY],
        additional_checks=[True, False]
    )


def ucdd_improved_write_to_file_submajority_withcheck_increasing_max_iters():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[10, 100, 500, 1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/all_numerical_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[10, 100, 500, 1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_exclude_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[10, 100, 500, 1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_target_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[10, 100, 500, 1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_onehot_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )


def ucdd_improved_write_to_file_submajority_withcheck_decreasing_tols():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-3, 1e-4, 5e-5, 1e-5],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/all_numerical_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-3, 1e-4, 5e-5, 1e-5],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_exclude_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-3, 1e-4, 5e-5, 1e-5],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_target_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[1e-3, 1e-4, 5e-5, 1e-5],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_onehot_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )


def ucdd_improved_write_to_file_submajority_withcheck_increasing_n_inits():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[50, 100, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/all_numerical_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[50, 100, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_exclude_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[50, 100, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_target_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[50, 100, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check/mixed_onehot_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )


def ucdd_improved_write_to_file_some_numerical_play_with_n_init_max_iter_and_tol():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths[:2],
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[100],
        tols=[100],
        result_file='ucdd_improved/results/all_numerical_results_very_high_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )


def ucdd_improved_write_to_file_submajority_withcheck_increasing_max_iters_new():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[2, 10, 50, 250],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/all_numerical_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[2, 10, 50, 250],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/mixed_exclude_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[2, 10, 50, 250],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/mixed_target_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[2, 10, 50, 250],
        tols=[1e-4],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/mixed_onehot_increasing_max_iter.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )


def ucdd_improved_write_to_file_submajority_withcheck_decreasing_tols_new():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[10, 1e-2, 1e-4, 1e-6],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/all_numerical_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[10, 1e-2, 1e-4, 1e-6],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/mixed_exclude_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[10, 1e-2, 1e-4, 1e-6],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/mixed_target_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[100],
        max_iters=[1000],
        tols=[10, 1e-2, 1e-4, 1e-6],
        min_runs=2,
        result_file='ucdd_improved/results_submajority_with_check2/mixed_onehot_decreasing_tol.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )


def ucdd_improved_write_to_file_submajority_withcheck_increasing_n_inits_new():
    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_numerical_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[2, 10, 50, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=10,  # min_runs are increased, because for very little n_inits, the results will be unstable
        result_file='ucdd_improved/results_submajority_with_check2/all_numerical_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.EXCLUDE],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[2, 10, 50, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=10,  # min_runs are increased, because for very little n_inits, the results will be unstable
        result_file='ucdd_improved/results_submajority_with_check2/mixed_exclude_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.TARGET],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[2, 10, 50, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=10,  # min_runs are increased, because for very little n_inits, the results will be unstable
        result_file='ucdd_improved/results_submajority_with_check2/mixed_target_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )

    ucdd_improved.ucdd_result_writer.eval_and_write_to_file(
        data_paths=only_mixed_data_paths,
        encodings=[ucdd_improved.ucdd_supported_parameters.Encoders.ONEHOT],
        test_fraction=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        true_drift_idx=2,
        n_inits=[2, 10, 50, 200],
        max_iters=[1000],
        tols=[1e-4],
        min_runs=10,  # min_runs are increased, because for very little n_inits, the results will be unstable
        result_file='ucdd_improved/results_submajority_with_check2/mixed_onehot_increasing_n_inits.csv',
        train_batch_strategies=[ucdd_improved.ucdd_supported_parameters.TrainBatchStrategies.SUBMAJORITY],
        additional_checks=[True]
    )
