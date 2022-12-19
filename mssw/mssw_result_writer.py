import csv
import os

import pandas as pd

import mssw.mssw_eval_local_datasets
from pathlib import Path

from mssw import mssw_eval_local_datasets


def eval_and_write(
        data_paths,
        encodings,
        test_fraction,
        num_ref_batches,
        num_test_batches,
        true_drift_idx,
        num_clusters=2,
        first_random_state=0,
        coeff=2.66,
        min_runs=10,
        std_err_threshold=0.05
):
    argument_results = mssw_eval_local_datasets.eval_multiple_parameter_sets(
        data_paths,
        encodings,
        test_fraction,
        num_ref_batches,
        num_test_batches,
        true_drift_idx,
        num_clusters,
        first_random_state,
        coeff,
        min_runs,
        std_err_threshold
    )

    print('argument_results')
    print(argument_results)

    for argument_result in argument_results:
        data_path = argument_result['data_path']
        runs_results_bool = argument_result['runs_results_bool']
        argument_result.pop('runs_results_bool')

        folder_directory = data_path.split('/')[1:]
        folder_directory[-1] = folder_directory[-1].split('.')[0]
        folder_directory = 'mssw/results_of_runs/' + '/'.join(folder_directory[:-1]) +\
                           '/' + folder_directory[-1] + '/' + argument_result['encoding'] + '_result.csv'
        print('folder_directory')
        print(folder_directory)
        path = Path(folder_directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            wr = csv.writer(f)
            # wr.writerow(['a', 'b', 'c'])
            wr.writerows(argument_result.items())
            wr.writerow(('num_runs', len(runs_results_bool)))
            wr.writerows(runs_results_bool)


def combine_synthetic_results():
    abrupt_path = Path('mssw/results_of_runs/synthetic_data/abrupt_drift')
    all_abrupt_folders = os.listdir(abrupt_path)
    final_result_dict = {
        'dataset': [], 'data': [], 'drift': [], 'width': [], 'encoding': [], 'FPR_mean': [], 'latency_mean': []
    }
    for abrupt_folder in all_abrupt_folders:
        abrupt_files_in_folder = os.listdir(abrupt_path.__str__() + '/' + abrupt_folder)
        for abrupt_file in abrupt_files_in_folder:
            full_file_path = abrupt_path.__str__() + '/' + abrupt_folder + '/' + abrupt_file
            with open(full_file_path) as f:
                rdr = csv.reader(f)
                result_dict = {}
                description = ''
                while description != 'num_runs':
                    two_element_line = rdr.__next__()
                    description = two_element_line[0]
                    value = two_element_line[1]
                    result_dict[description] = value
                print('result dict')
                print(result_dict)

                final_result_dict['dataset'].append(abrupt_folder.split('_')[0])
                final_result_dict['data'].append('synthetic')
                final_result_dict['drift'].append('abrupt')
                final_result_dict['width'].append(0)
                final_result_dict['encoding'].append(result_dict['encoding'])
                final_result_dict['FPR_mean'].append(result_dict['fpr_mean'])
                final_result_dict['latency_mean'].append(result_dict['latency_mean'])

    gradual_path = Path('mssw/results_of_runs/synthetic_data/gradual_drift')
    all_gradual_folders = os.listdir(gradual_path)
    for gradual_folder in all_gradual_folders:
        gradual_files_in_folder = os.listdir(gradual_path.__str__() + '/' + gradual_folder)
        for gradual_file in gradual_files_in_folder:
            full_file_path = gradual_path.__str__() + '/' + gradual_folder + '/' + gradual_file
            with open(full_file_path) as f:
                rdr = csv.reader(f)
                result_dict = {}
                description = ''
                while description != 'num_runs':
                    two_element_line = rdr.__next__()
                    description = two_element_line[0]
                    value = two_element_line[1]
                    result_dict[description] = value
                print('result dict')
                print(result_dict)

                final_result_dict['dataset'].append(gradual_folder.split('_')[0])
                final_result_dict['data'].append('synthetic')
                final_result_dict['drift'].append('gradual')
                drift_width = gradual_folder.split('_')[-1]
                drift_width = 0.5 if drift_width == '05' else float(drift_width)
                final_result_dict['width'].append(drift_width)
                final_result_dict['encoding'].append(result_dict['encoding'])
                final_result_dict['FPR_mean'].append(float(result_dict['fpr_mean']))
                final_result_dict['latency_mean'].append(float(result_dict['latency_mean']))

    final_result_df = pd.DataFrame.from_dict(final_result_dict)
    print('final result_df')
    print(final_result_df)

    sorted_final_result_df = final_result_df.sort_values(['drift', 'dataset', 'encoding', 'width'])
    print('sorted')
    print(sorted_final_result_df)

    path = 'mssw/mssw_final_result.csv'
    sorted_final_result_df.to_csv(path, index=False)
