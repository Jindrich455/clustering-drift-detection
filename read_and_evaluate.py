import csv

import numpy as np


def csv_to_2d_str_list(csv_path):
    results = []
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            results.append(row)
    return results


def raw_results_2d_int_list(csv_path):
    results = csv_to_2d_str_list(csv_path)
    results = [[i == 'True' for i in res] for res in results]
    results = np.array(results)
    results_int = []
    for i, res in enumerate(results):
        int_arr = np.nonzero(res)[0]
        results_int.append(int_arr.tolist())

    return results_int


