import csv
import os
import random

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

import accepting
import my_preprocessing
import ucdd
import ucdd_eval
import ucdd_eval_and_write_res
import ucdd_visual_inspection
import supported_parameters as spms


def take_random_states(num_runs):
    with open('runs_results/random_states.csv') as f:
        rd = csv.reader(f)
        all_random_states = list(map(int, rd.__next__()))
    random_states = all_random_states[:num_runs]
    return random_states


def generate_random_states():
    initial_random_state = 3
    random.seed(initial_random_state)
    random_states = random.sample(range(0, 10000), 1000)

    with open('runs_results/random_states.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(random_states)


def write_detections_to_file(all_detections, path, filename):
    # print('current # files', str(len(os.listdir('runs_results/' + path))))
    # current_n_files = len(os.listdir('runs_results/' + path))
    # filename = 'run' + str(current_n_files + 1) + '.csv'
    # print('filename', filename)
    with open('runs_results/' + path + '/' + filename, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(all_detections)


def convert_filenames_to_directories():
    files = os.listdir('runs_results/synthetic_data/gradual_drift')
    print(files)
    raw_names = []
    for filename in files:
        raw_names.append(filename[:-5])
    for raw_name in raw_names:
        os.mkdir('runs_results/synthetic_data/gradual_drift/' + raw_name)


if __name__ == '__main__':
    # random_states = take_random_states(num_runs=2)

    # all_occurrences = ucdd_eval.evaluate_ucdd_multiple_random_states(
    #     file_path='Datasets_concept_drift/' + path + '.arff',
    #     scaling=spms.Scalers.MINMAX,
    #     encoding=spms.Encoders.EXCLUDE,
    #     test_size=0.7,
    #     num_ref_batches=3,
    #     num_test_batches=7,
    #     random_states=random_states,
    #     additional_check=False,
    #     detect_all_training_batches=False,
    #     use_pyclustering=True,
    #     metric_id=spms.Distances.EUCLIDEAN
    # )

    # ucdd_eval_and_write_res.eval_and_write(
    #     rel_path='synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced',
    #     scaling=spms.Scalers.MINMAX,
    #     encoding=spms.Encoders.EXCLUDE,
    #     test_size=0.7,
    #     num_ref_batches=3,
    #     num_test_batches=7,
    #     num_runs=10,
    #     additional_check=False,
    #     detect_all_training_batches=False,
    #     use_pyclustering=True,
    #     metric_id=spms.Distances.EUCLIDEAN
    # )

    ucdd_eval_and_write_res.eval_and_write_all(
        rel_path='synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced',
        scaling_list=[spms.Scalers.MINMAX],
        encoding_list=[spms.Encoders.EXCLUDE],
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        num_runs=10,
        additional_check_list=[True, False],
        detect_all_training_batches_list=[False],
        use_pyclustering_list=[True],
        metric_id_list=[spms.Distances.EUCLIDEAN]
    )

    # print('all_occurrences', all_occurrences)
    #
    # write_detections_to_file(all_occurrences, path,
    #                          filename='no_encoding_with_minmax_euclidean_no_check_no_alltraining_10_runs' + '.csv')

    # ucdd_eval.evaluate_ucdd(
    #     file_path='tests/test_datasets/drift_2d.arff',
    #     scaling="minmax",
    #     encoding="none",
    #     test_size=0.5,
    #     num_ref_batches=1,
    #     num_test_batches=1,
    #     random_state=0,
    #     additional_check=True,
    #     use_pyclustering=True,
    #     debug=True
    # )

    # ucdd_visual_inspection.show_ucdd()


    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    #     ('cat', OneHotEncoder(sparse=False), selector(dtype_exclude='number'))
    # ])
    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    #     ('cat', TargetEncoder(), selector(dtype_exclude='number'))
    # ])
    # transformer = FunctionTransformer(lambda x: x)
    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    # ])
