import os
import time

import numpy as np
import pyclustering.utils
import scipy.spatial.distance
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import type_metric, distance_metric

from ucdd import ucdd_supported_parameters as spms, ucdd_eval_and_write_res, ucdd_eval, ucdd_read_and_evaluate


def big_evaluation0():
    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=['Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff'],
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True],
        detect_all_training_batches_list=[False],
        metric_ids=[spms.Distances.EUCLIDEAN],
        use_pyclustering=True
    )


def big_evaluation():
    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=['Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff'],
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True, False],
        metric_ids=[spms.Distances.EUCLIDEAN, spms.Distances.MANHATTAN],
        use_pyclustering=True
    )


def big_evaluation2():
    ucdd_eval_and_write_res.eval_and_write_all(
        dataset_paths=[
            'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced.arff',
            'Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff'
        ],
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE, spms.Encoders.ONEHOT, spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True, False],
        metric_ids=[spms.Distances.EUCLIDEAN, spms.Distances.MANHATTAN],
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
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[False],
        metric_ids=[spms.Distances.EUCLIDEAN, spms.Distances.MANHATTAN],
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
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE, spms.Encoders.ONEHOT, spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[False],
        detect_all_training_batches_list=[False],
        metric_ids=[spms.Distances.EUCLIDEAN],
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
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE, spms.Encoders.ONEHOT, spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True],
        detect_all_training_batches_list=[False],
        metric_ids=[spms.Distances.EUCLIDEAN],
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
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE, spms.Encoders.ONEHOT, spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[False],
        metric_ids=[spms.Distances.MANHATTAN],
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
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE, spms.Encoders.ONEHOT, spms.Encoders.TARGET],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True],
        metric_ids=[spms.Distances.EUCLIDEAN, spms.Distances.MANHATTAN],
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
        scalings=[spms.Scalers.MINMAX],
        encodings=[spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        additional_checks=[True, False],
        detect_all_training_batches_list=[True],
        metric_ids=[spms.Distances.EUCLIDEAN, spms.Distances.MANHATTAN],
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
        spms.Scalers.MINMAX,
        spms.Encoders.ONEHOT,
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

