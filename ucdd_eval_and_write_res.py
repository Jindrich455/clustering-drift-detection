import csv
import os
import random
import itertools

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

import accepting
import my_preprocessing
import ucdd
import ucdd_eval
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


def filename_from_params(
        encoding,
        scaling,
        metric_id,
        additional_check,
        detect_all_training_batches,
        num_random_states,
        num_test_batches):
    filename = ''
    filename += encoding.name.lower() + '_encoding'
    filename += '_' + scaling.name.lower() + '_scaling'
    filename += '_' + metric_id.name.lower() + '_distance'
    filename += '_' + ('with_check' if additional_check else 'no_check')
    filename += '_' + ('with_alltraining' if detect_all_training_batches else 'no_alltraning')
    filename += '_' + str(num_random_states) + '_runs'
    filename += '_' + str(num_test_batches) + '_tbs'
    return filename


def write_metrics_to_file(mean_fpr, mean_latency, relative_path, filename):
    with open('runs_results/' + relative_path + '/' + filename, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([mean_fpr])
        wr.writerow([mean_latency])


def eval_and_write(
        dataset_path,
        scaling,
        encoding,
        test_size,
        num_ref_batches,
        num_test_batches,
        additional_check,
        detect_all_training_batches,
        metric_id,
        use_pyclustering=True
):
    all_occurrences, mean_fpr, mean_latency = ucdd_eval.evaluate_ucdd_until_convergence(
        file_path=dataset_path,
        scaling=scaling,
        encoding=encoding,
        test_size=test_size,
        num_ref_batches=num_ref_batches,
        num_test_batches=num_test_batches,
        additional_check=additional_check,
        detect_all_training_batches=detect_all_training_batches,
        use_pyclustering=use_pyclustering,
        metric_id=metric_id
    )

    filename = filename_from_params(encoding, scaling, metric_id, additional_check,
                                    detect_all_training_batches,
                                    len(all_occurrences),
                                    num_test_batches)

    relative_path = '/'.join(dataset_path.split('.')[0].split('/')[1:])

    print('filename', filename)
    all_occurrences_binary = []
    for drift_locations in all_occurrences:
        all_current_occurrences_binary = np.repeat(False, num_test_batches)
        all_current_occurrences_binary[drift_locations] = True
        all_occurrences_binary.append(all_current_occurrences_binary)
    print('all_occurrences_binary', all_occurrences_binary)
    write_detections_to_file(all_occurrences_binary, relative_path,
                             filename=filename + '_raw.csv')
    write_metrics_to_file(mean_fpr, mean_latency, relative_path,
                          filename=filename + '_metrics.csv')


def eval_and_write_all(
        dataset_path,
        scalings,
        encodings,
        test_size,
        num_ref_batches,
        num_test_batches,
        additional_checks,
        detect_all_training_batches_list,
        metric_ids,
        use_pyclustering=True
):

    arg_tuples = list(itertools.product(scalings, encodings, additional_checks,
                                        detect_all_training_batches_list, metric_ids))

    for i, arg_tuple in enumerate(arg_tuples):
        print('argument combination #', i)
        scaling = arg_tuple[0]
        encoding = arg_tuple[1]
        additional_check = arg_tuple[2]
        detect_all_training_batches = arg_tuple[3]
        metric_id = arg_tuple[4]

        print(scaling)
        print(encoding)
        print(additional_check)
        print(detect_all_training_batches)
        print(use_pyclustering)
        print(metric_id)

        eval_and_write(
            dataset_path,
            scaling,
            encoding,
            test_size,
            num_ref_batches,
            num_test_batches,
            additional_check,
            detect_all_training_batches,
            metric_id
        )
