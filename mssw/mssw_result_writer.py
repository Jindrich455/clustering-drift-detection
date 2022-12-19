import csv

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

    for i, data_path in enumerate(data_paths):
        argument_result = argument_results[i]
        runs_results_bool = argument_result['runs_results_bool']
        argument_result.pop('runs_results_bool')

        folder_directory = data_path.split('/')[1:]
        folder_directory[-1] = folder_directory[-1].split('.')[0]
        folder_directory = 'mssw/results_of_runs/' + '/'.join(folder_directory[:-1]) +\
                           '/' + folder_directory[-1] + '_result.csv'
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


