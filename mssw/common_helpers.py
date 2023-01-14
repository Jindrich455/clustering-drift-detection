import math
import numpy as np
import csv
import sys

from sklearn.cluster import KMeans


def split_to_fixed_size_batches(X, y, batch_size):
    """Split X and y to batches of the given batch_size"""
    chunk_size = batch_size
    print('chunk size', chunk_size)

    num_chunks = math.ceil(X.shape[0] / chunk_size)
    print('number of chunks', num_chunks)
    print('number of data', X.shape[0])
    X_batches = np.array_split(X, num_chunks)
    y_batches = np.array_split(y, num_chunks)

    print('number of resulting batches', len(X_batches))
    print(X_batches[0])
    print(X_batches[0].shape)

    return X_batches, y_batches


def write_verbose_kmeans_to_file(result_filename, data_to_cluster, n_clusters, n_init, max_iter, tol, random_state):
    print('random state:', random_state)
    orig_stdout = sys.stdout
    sys.stdout = open(result_filename, 'wt')

    fitted_kmeans = KMeans(
        n_clusters=2,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=3,
        random_state=random_state
    ).fit(data_to_cluster)

    sys.stdout = orig_stdout


print('something')


def convert_kmeans_output_file_to_dicts(file_path, n_init):
    # read the verbose output file to be able to reach conclusions
    with open(file_path, newline='') as f:
        rdr = csv.reader(f)
        kmeans_verbose_output_list = list(rdr)

    run_results_list = []
    reversed_run_results_list = []

    current_run_reversed_list_messages = []
    for el in reversed(kmeans_verbose_output_list):
        current_run_reversed_list_messages.append(el)
        if el[0] == 'Initialization complete':
            reversed_run_results_list.append(current_run_reversed_list_messages)
            current_run_reversed_list_messages = []

    for reversed_run_result in reversed_run_results_list:
        run_result = reversed_run_result.copy()
        run_result.reverse()
        run_results_list.append(run_result)

    run_results_list.reverse()

    run_results_dicts = []
    for i in range(n_init):
        result_dict = {'converged': False, 'convergence_dict': {}, 'iterations_inertia': []}
        current_result = run_results_list[i]
        only_iterations = current_result[1:]
        if len(current_result[-1]) == 1:  # this run converged
            result_dict['converged'] = True
            only_iterations = only_iterations[:-1]
            converge_message_split = current_result[-1][0].split(' ')
            result_dict['convergence_dict']['iteration'] = int(converge_message_split[3][:-1])
            if len(converge_message_split) == 6:
                result_dict['convergence_dict']['type'] = 'strict'
            else:
                result_dict['convergence_dict']['type'] = 'tol-based'
                result_dict['convergence_dict']['center_shift'] = converge_message_split[6]
                result_dict['convergence_dict']['within_tol'] = converge_message_split[9]

        iterations_inertia = []
        for iteration_message_list in only_iterations:
            inertia = float(iteration_message_list[1].split(' ')[2][:-1])
            iterations_inertia.append(inertia)
        result_dict['iterations_inertia'] = iterations_inertia

        run_results_dicts.append(result_dict)

    return run_results_dicts


def print_stats_from_kmeans_output_dicts(run_results_dicts):
    max_iterations = -1
    initial_inertia = []
    final_inertia = []
    num_convergences = 0
    num_strict_convergences = 0
    num_tol_based_convergences = 0
    for result_dict in run_results_dicts:
        num_iterations = len(result_dict['iterations_inertia'])
        max_iterations = num_iterations if num_iterations > max_iterations else max_iterations
        initial_inertia.append(result_dict['iterations_inertia'][0])
        final_inertia.append(result_dict['iterations_inertia'][-1])
        if result_dict['converged'] == True:
            num_convergences += 1
            if result_dict['convergence_dict']['type'] == 'strict':
                num_strict_convergences += 1
            else:
                num_tol_based_convergences += 1

    print('total number of results:', len(run_results_dicts))
    print('maximum number of iterations:', max_iterations)
    print('minimum initial inertia:', min(initial_inertia))
    print('maximum initial inertia:', max(initial_inertia))
    print('number of unique final inertia values:', len(np.unique(final_inertia)))
    print('minimum final inertia:', min(final_inertia))
    print('maximum final inertia:', max(final_inertia))
    print('total number of convergences:', num_convergences)
    print('number of strict convergences:', num_strict_convergences)
    print('number of tol-based convergences:', num_tol_based_convergences)
