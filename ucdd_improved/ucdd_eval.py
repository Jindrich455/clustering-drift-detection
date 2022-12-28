import numpy as np

import mssw.mssw
import ucdd_improved.ucdd


def fpr_and_latency_when_averaging(drift_locations, num_test_batches, true_drift_idx):
    """The inputs drift_locations and true_drift_idx are is zero-indexed"""
    fpr = 0
    latency = 1
    drift_locations_arr = np.array(drift_locations)
    signal_locations_before_drift = drift_locations_arr[drift_locations_arr < true_drift_idx]
    signal_locations_not_before_drift = drift_locations_arr[drift_locations_arr >= true_drift_idx]
    num_batches_after_first_drift = num_test_batches - (true_drift_idx + 1)
    drift_detected = False # says whether some drift detection was triggered at or after a drift occurrence

    if len(drift_locations) >= 1:
        if len(signal_locations_before_drift) > 0:
            fpr = len(signal_locations_before_drift) / true_drift_idx
        if len(signal_locations_not_before_drift) > 0:
            first_useful_drift_signal = signal_locations_not_before_drift[0]
            latency = (first_useful_drift_signal - true_drift_idx) / num_batches_after_first_drift
            drift_detected = True

    return fpr, latency, drift_detected


def all_drifting_batches_randomness_robust(reference_data_batches, testing_data_batches, train_batch_strategy,
                                           additional_check,
                                           n_init=10,
                                           max_iter=300, tol=1e-4, true_drift_idx=2, first_random_state=0,
                                           min_runs=10, std_err_threshold=0.05):
    """
    Repeat running ucdd_improved.ucdd.all_drifting_batches(...) until the s.e. of metrics from different runs is low enough

    :param n_init:
    :param max_iter:
    :param tol:
    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param n_clusters: desired number of clusters for kmeans
    :param first_random_state: random states used will be incremented from this one
    :param coeff: coeff used to detect drift, default=2.66
    :param std_err_threshold: threshold to stop executing the mssw algorithm
    :return: a list of lists from all_drifting_batches(...), and the mean and s.e. of FPR and latency
    """
    fprs = []
    latencies = []
    runs_results_bool = []
    fpr_std_err = -1
    latency_std_err = -1
    num_runs = 0
    random_state = first_random_state
    while num_runs < min_runs or max(fpr_std_err, latency_std_err) > std_err_threshold:
        drifting_batches_bool = ucdd_improved.ucdd.all_drifting_batches(
            reference_data_batches,
            testing_data_batches,
            train_batch_strategy=train_batch_strategy,
            additional_check=additional_check,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        # print('drifting_batches_bool')
        # print(drifting_batches_bool)
        drift_locations = np.arange(len(drifting_batches_bool))[drifting_batches_bool]
        # print('drift_locations')
        # print(drift_locations)
        fpr, latency, _ = fpr_and_latency_when_averaging(
            drift_locations,
            len(testing_data_batches),
            true_drift_idx
        )
        fprs.append(fpr)
        latencies.append(latency)
        runs_results_bool.append(drifting_batches_bool)
        num_runs += 1
        random_state += n_init

        # print('number of runs', num_runs)
        if num_runs >= min_runs:
            fpr_std_err = np.std(fprs) / np.sqrt(len(fprs))
            latency_std_err = np.std(latencies) / np.sqrt(len(latencies))
        # print('fprs', fprs, 's.e.', fpr_std_err)
        # print('latencies', latencies, 's.e.', latency_std_err)

    final_fpr_mean = np.mean(fprs)
    final_latency_mean = np.mean(latencies)
    return runs_results_bool, final_fpr_mean, fpr_std_err, final_latency_mean, latency_std_err
