import pandas as pd

import ucdd.ucdd_eval_local_datasets


def eval_and_write_to_file(data_paths, encodings, test_fraction, num_ref_batches, num_test_batches,
                           true_drift_idx,
                           result_file,
                           train_batch_strategies, additional_checks,
                           n_inits=[10], max_iters=[300], tols=[1e-4], first_random_state=0,
                           min_runs=10, std_err_threshold=0.05
                           ):
    argument_results = ucdd.ucdd_eval_local_datasets.eval_multiple_parameter_sets(
        data_paths, encodings, test_fraction, num_ref_batches, num_test_batches,
        true_drift_idx,
        train_batch_strategies, additional_checks,
        n_inits=n_inits, max_iters=max_iters, tols=tols, first_random_state=first_random_state,
        min_runs=min_runs, std_err_threshold=std_err_threshold
    )

    print('argument_results')
    print(argument_results)

    final_result_dict = {
        'type_of_data': [], 'dataset': [], 'drift': [], 'width': [], 'encoding': [],
        'train_batch_strategy': [], 'additional_check': [],
        'n_init': [], 'max_iter': [], 'tol': [],
        'FPR_mean': [], 'latency_mean': []
    }

    for argument_result in argument_results:
        data_path = argument_result['data_path']
        runs_results_bool = argument_result['runs_results_bool']
        argument_result.pop('runs_results_bool')

        data_filename = data_path.split('/')[-1]
        type_of_data = data_path.split('/')[1].split('_')[0]  # synthetic or real-world
        dataset_name = data_filename.split('_')[0]  # sea, agraw1, agraw2
        drift_type = data_path.split('/')[2].split('_')[0]
        drift_width = '0' if drift_type == 'abrupt' else data_filename.split('_')[-1].split('.')[0]
        drift_width = 0.5 if drift_width == '05' else float(drift_width)
        encoding = argument_result['encoding']
        train_batch_strategy = argument_result['train_batch_strategy']
        additional_check = argument_result['additional_check']
        n_init = argument_result['n_init']
        max_iter = argument_result['max_iter']
        tol = argument_result['tol']
        fpr_mean = float(argument_result['fpr_mean'])
        latency_mean = float(argument_result['latency_mean'])

        final_result_dict['type_of_data'].append(type_of_data)
        final_result_dict['dataset'].append(dataset_name)
        final_result_dict['drift'].append(drift_type)
        final_result_dict['width'].append(drift_width)
        final_result_dict['encoding'].append(encoding)
        final_result_dict['train_batch_strategy'].append(train_batch_strategy)
        final_result_dict['additional_check'].append(additional_check)
        final_result_dict['n_init'].append(n_init)
        final_result_dict['max_iter'].append(max_iter)
        final_result_dict['tol'].append(tol)
        final_result_dict['FPR_mean'].append(fpr_mean)
        final_result_dict['latency_mean'].append(latency_mean)

    print('final result dict')
    print(final_result_dict)

    final_result_df = pd.DataFrame.from_dict(final_result_dict)
    print('final result df')
    print(final_result_df)

    sorted_final_result_df = final_result_df.sort_values(['drift', 'dataset', 'encoding', 'width'])
    print('sorted')
    print(sorted_final_result_df)

    sorted_final_result_df.to_csv(result_file, index=False)