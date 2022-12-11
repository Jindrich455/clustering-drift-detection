import csv
import os

import numpy as np
import pandas as pd

from ucdd import ucdd_eval

from pathlib import Path


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


def info_from_filename(filename):
    info_list = filename.split('_')[:-1]
    names = info_list[1::2]
    values = info_list[::2]
    # encoding_str = info_list[0]
    # scaling_str = info_list[2]
    # distance_str = info_list[4]
    # check_str = 'no' if info_list[6] == 'no' else 'yes'
    # alltraining_str = 'no' if info_list[8] == 'no' else 'yes'
    # num_runs = info_list[10]

    info_tuples = zip(names, values)
    info_dict = dict((x, y) for x, y in info_tuples)

    return info_dict


def df_from_filename(filename):
    info_dict = info_from_filename(filename)
    # print(info_dict)
    info_df = pd.DataFrame(info_dict, index=[0])
    print('filename_info_df')
    print(info_df.to_string())

    return info_df


def metrics_from_results(results_int):
    fprs_for_avg = []
    latencies_for_avg = []
    drifts_detected = []
    for res in results_int:
        fpr_for_avg, latency_for_avg, drift_detected = ucdd_eval.fpr_and_latency_when_averaging(
            res, num_test_batches=7, true_drift_idx=2)
        fprs_for_avg.append(fpr_for_avg)
        latencies_for_avg.append(latency_for_avg)
        drifts_detected.append(drift_detected)
    # print(fprs_for_avg)
    # print(latencies_for_avg)
    # print(drifts_detected)
    mean_fpr = round(np.mean(fprs_for_avg), 2)
    mean_latency = round(np.mean(latencies_for_avg), 2)
    drift_detection_rate = round(np.mean(drifts_detected), 2)

    # print('mean_fpr', mean_fpr)
    # print('mean_latency', mean_latency)
    # print('drift_detection_rate', drift_detection_rate)

    metrics_df = pd.DataFrame({'detection_rate': drift_detection_rate, 'mean_FPR': mean_fpr, 'mean_latency': mean_latency}, index=[0])
    print('metrics_df')
    print(metrics_df.to_string())

    return metrics_df


def all_info_for_file(csv_path):
    results = raw_results_2d_int_list(csv_path)
    metrics_df = metrics_from_results(results)

    filename = csv_path.split('/')[-1]
    filename_df = df_from_filename(filename)

    all_info_df = filename_df.join(metrics_df)
    print('all_info_df')
    print(all_info_df.to_string())

    return all_info_df


def all_info_for_files(csv_paths):
    final_df = pd.DataFrame()
    print('initial final_df')
    print(final_df.to_string())
    all_info_dfs = []
    for csv_path in csv_paths:
        all_info_df = all_info_for_file(csv_path)
        print('all_info_df columns')
        print(all_info_df.columns)
        print('all_info_df index')
        print(all_info_df.index)
        # # all_info_df = all_info_df.set_index(pd.Int64Index([final_df.shape[0] + 1]))
        # print('all_info_df reindexed')
        # print(all_info_df.to_string())
        # final_df = pd.concat([final_df, all_info_df], ignore_index=True)
        # print('final_df')
        # print(final_df.to_string())
        all_info_dfs.append(all_info_df)

    final_df = pd.concat(all_info_dfs)
    print('final_df')
    print(final_df.to_string())

    return final_df


def all_info_for_all_files_in_folder(folder_path):
    filenames_raw_results = []
    with os.scandir(folder_path) as files:
        for file in files:
            filename = file.name
            filetype = filename.split('_')[-1].split('.')[0]
            if filetype == 'raw':
                filenames_raw_results.append(folder_path + '/' + filename)

    print(filenames_raw_results)
    all_info_df = all_info_for_files(filenames_raw_results)
    print(all_info_df)
    return all_info_df


def rows_with_drift_detected(all_info_df):
    useful_info_df = all_info_df[all_info_df['detection_rate'] > 0]
    print('only useful rows')
    print(useful_info_df.to_string())
    return useful_info_df


def write_all_info_df_to_csv(raw_results_csv_path, all_info_df):
    path = Path('ucdd/overall_results/' + '/'.join(raw_results_csv_path.split('/')[2:-1]) + '/metrics.csv')
    print('path', path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_info_df.to_csv(path, index=False)


def write_all_info_all_files_df_to_csv(folder_path, all_info_df, new_file_name):
    path = Path('ucdd/overall_results/' + '/'.join(folder_path.split('/')[2:]) + '/' + new_file_name)
    print('path', path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_info_df.to_csv(path, index=False)

