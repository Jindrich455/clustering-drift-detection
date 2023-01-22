import math
import numpy as np
import csv
import sys

from sklearn.cluster import KMeans


def write_kmeans_results_ucdd_helper(output_filename_no_extension, ref_batches, n_init, max_iter, tol, random_state,
                                     show_all=False):
    # dummy = [np.asarray(1), np.asarray(2), np.asarray(3)]
    combinations = []
    for i in range(len(ref_batches)):
    #     combinations.append(np.vstack((dummy[i], dummy[(i + 1) % 3])))
        combinations.append(np.vstack((ref_batches[i], ref_batches[(i + 1) % 3])))

    all_results_from_combinations = []
    for i, combination in enumerate(combinations):
        filename = output_filename_no_extension + str(i) + '.txt'
        print('filename', filename)
        write_verbose_kmeans_to_file(result_filename=output_filename_no_extension + str(i) + '.txt',
                                     data_to_cluster=combination,
                                     n_clusters=2, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
        output_dicts = convert_kmeans_output_file_to_dicts(filename, n_init=n_init)
        all_results_from_combinations.append(output_dicts)
        if show_all:
            print_stats_from_kmeans_output_dicts(output_dicts)

    print_stats_from_all_combinations(all_results_from_combinations)


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


def get_stats_from_kmeans_output_dicts(run_results_dicts):
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

    return {
        'total_num_results': len(run_results_dicts),
        'max_iterations': max_iterations,
        'min_init_inertia': min(initial_inertia),
        'max_init_inertia': max(initial_inertia),
        'num_unique_final_inertia': len(np.unique(final_inertia)),
        'min_final_inertia': min(final_inertia),
        'max_final_inertia': max(final_inertia),
        'total_num_convergences': num_convergences,
        'num_strict_convergences': num_strict_convergences,
        'num_tol_based_convergences': num_tol_based_convergences
    }


def print_stats_from_kmeans_output_dicts(run_results_dicts):
    stats_from_dicts = get_stats_from_kmeans_output_dicts(run_results_dicts)

    print('total number of results:', stats_from_dicts['total_num_results'])
    print('maximum number of iterations:', stats_from_dicts['max_iterations'])
    print('minimum initial inertia:', stats_from_dicts['min_init_inertia'])
    print('maximum initial inertia:', stats_from_dicts['max_init_inertia'])
    print('number of unique final inertia values:', stats_from_dicts['num_unique_final_inertia'])
    print('minimum final inertia:', stats_from_dicts['min_final_inertia'])
    print('maximum final inertia:', stats_from_dicts['max_final_inertia'])
    print('total number of convergences:', stats_from_dicts['total_num_convergences'])
    print('number of strict convergences:', stats_from_dicts['num_strict_convergences'])
    print('number of tol-based convergences:', stats_from_dicts['num_tol_based_convergences'])


def print_stats_from_all_combinations(all_results_from_combinations):
    final_stats_dict = {
        'total_max_iterations': 0,
        'total_min_init_inertia': 0,
        'total_max_init_inertia': 0,
        'total_min_final_inertia': 0,
        'total_max_final_inertia': 0,
        'total_num_convergences': 0,
        'total_num_strict_convergences': 0,
        'total_num_tol_based_convergences': 0
    }
    # total_max_iterations = 0
    # total_min_init_inertia = 0
    # total_max_init_inertia = 0
    # total_min_final_inertia = 0
    # total_max_final_inertia = 0
    # total_num_convergences = 0
    # total_num_strict_convergences = 0
    # total_num_tol_based_convergences = 0
    for result_from_combination in all_results_from_combinations:
        relevant_stats = get_stats_from_kmeans_output_dicts(result_from_combination)
        final_stats_dict['total_max_iterations'] = max(final_stats_dict['total_max_iterations'],
                                                       relevant_stats['max_iterations'])
        final_stats_dict['total_min_init_inertia'] =\
            min(final_stats_dict['total_min_init_inertia'], relevant_stats['min_init_inertia']) \
                if final_stats_dict['total_min_init_inertia'] > 0\
                else relevant_stats['min_init_inertia']
        final_stats_dict['total_max_init_inertia'] = max(final_stats_dict['total_max_init_inertia'],
                                                         relevant_stats['max_init_inertia'])
        final_stats_dict['total_min_final_inertia'] =\
            min(final_stats_dict['total_min_final_inertia'], relevant_stats['min_final_inertia'])\
                if final_stats_dict['total_min_final_inertia'] > 0\
                else relevant_stats['min_final_inertia']
        final_stats_dict['total_max_final_inertia'] = max(final_stats_dict['total_max_final_inertia'],
                                                          relevant_stats['max_final_inertia'])
        final_stats_dict['total_num_convergences'] += relevant_stats['total_num_convergences']
        final_stats_dict['total_num_strict_convergences'] += relevant_stats['num_strict_convergences']
        final_stats_dict['total_num_tol_based_convergences'] += relevant_stats['num_tol_based_convergences']

    print(final_stats_dict)


