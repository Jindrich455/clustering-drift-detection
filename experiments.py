import os

import read_and_evaluate
import ucdd_eval_and_write_res
import supported_parameters as spms


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


def save_metrics():
    path = 'runs_results/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = read_and_evaluate.all_info_for_all_files_in_folder(path)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = read_and_evaluate.rows_with_drift_detected(all_info_df)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_metrics2():
    path = 'runs_results/synthetic_data/abrupt_drift/agraw1_1_abrupt_drift_0_noise_balanced'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = read_and_evaluate.all_info_for_all_files_in_folder(path)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = read_and_evaluate.rows_with_drift_detected(all_info_df)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_metrics3():
    path = 'runs_results/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = read_and_evaluate.all_info_for_all_files_in_folder(path)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = read_and_evaluate.rows_with_drift_detected(all_info_df)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_metrics4():
    path = 'runs_results/synthetic_data/gradual_drift/sea_1_gradual_drift_0_noise_balanced_1'
    # all_info = read_and_evaluate.all_info_for_file(csv_path)
    # print('all_info')
    # print(all_info)
    # read_and_evaluate.write_all_info_df_to_csv(csv_path, all_info)
    # read_and_evaluate.all_info_for_files([csv_path1, csv_path2])
    all_info_df = read_and_evaluate.all_info_for_all_files_in_folder(path)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
    useful_info_df = read_and_evaluate.rows_with_drift_detected(all_info_df)
    read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')


def save_all_metrics():
    dir_path = 'runs_results/synthetic_data/gradual_drift'
    directory_names = os.listdir(dir_path)
    for name in directory_names:
        path = dir_path + '/' + name
        all_info_df = read_and_evaluate.all_info_for_all_files_in_folder(path)
        read_and_evaluate.write_all_info_all_files_df_to_csv(path, all_info_df, 'metrics.csv')
        useful_info_df = read_and_evaluate.rows_with_drift_detected(all_info_df)
        read_and_evaluate.write_all_info_all_files_df_to_csv(path, useful_info_df, 'useful_metrics.csv')



