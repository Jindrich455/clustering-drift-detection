import numpy as np

from core.ucdd import all_drifting_batches


def real_world_metrics(drift_signaled_bool, true_drift_bool):
    num_testing_batches = len(drift_signaled_bool)

    print(drift_signaled_bool)
    print(true_drift_bool)

    drift_correctly_signaled_bool = np.array(drift_signaled_bool) & np.array(true_drift_bool)
    drift_incorrectly_signaled_bool = np.array(drift_signaled_bool) & ~np.array(true_drift_bool)

    num_total_drifts = np.count_nonzero(true_drift_bool)
    num_total_non_drifts = num_testing_batches - num_total_drifts
    num_drift_correct_signals = np.count_nonzero(drift_correctly_signaled_bool)
    num_drift_incorrect_signals = np.count_nonzero(drift_incorrectly_signaled_bool)

    fpr = num_drift_incorrect_signals / num_total_non_drifts
    detection_accuracy = num_drift_correct_signals / num_total_drifts

    return fpr, detection_accuracy


def all_drifting_batches_randomness_robust(reference_data_batches, testing_data_batches,
                                           true_drift_bool,
                                           min_ref_batches_drift,
                                           additional_check,
                                           n_init=10,
                                           max_iter=300, tol=1e-4, first_random_state=0,
                                           min_runs=10, std_err_threshold=0.05,
                                           parallel=True):
    """
    Repeat running mssw.mssw.all_drifting_batches(...) until the s.e. of metrics from different runs is low enough

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
    print('min_runs', min_runs)

    fprs = []
    detection_accuracies = []
    runs_results_bool = []
    fpr_std_err = -1
    detection_accuracy_std_err = -1
    num_runs = 0
    random_state = first_random_state
    while num_runs < min_runs or max(fpr_std_err, detection_accuracy_std_err) > std_err_threshold:
        signaled_batches_bool = all_drifting_batches(
            reference_data_batches,
            testing_data_batches,
            min_ref_batches_drift=min_ref_batches_drift,
            additional_check=additional_check,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            parallel=parallel
        )

        # print('drifting_batches_bool')
        # print(drifting_batches_bool)

        fpr, detection_accuracy = real_world_metrics(signaled_batches_bool, true_drift_bool)

        fprs.append(fpr)
        detection_accuracies.append(detection_accuracy)
        runs_results_bool.append(signaled_batches_bool)
        num_runs += 1
        random_state += n_init

        # print('number of runs', num_runs)
        if num_runs >= min_runs:
            fpr_std_err = np.std(fprs) / np.sqrt(len(fprs))
            detection_accuracy_std_err = np.std(detection_accuracies) / np.sqrt(len(detection_accuracies))
        # print('fprs', fprs, 's.e.', fpr_std_err)
        # print('latencies', latencies, 's.e.', latency_std_err)

    final_fpr_mean = np.mean(fprs)
    final_detection_accuracy_mean = np.mean(detection_accuracies)
    return runs_results_bool, final_fpr_mean, fpr_std_err, final_detection_accuracy_mean, detection_accuracy_std_err


def results_multiple_min_ref_batches_drift(mrbd_vals, one_run_all_2d_drifts, true_drift_bool):
    fprs = []
    accs = []
    for mrbd in mrbd_vals:
        drifts_signaled_ref_batches_perc = np.mean(one_run_all_2d_drifts, axis=0)
        test_batch_drift_signals = drifts_signaled_ref_batches_perc > mrbd
        fpr, acc = real_world_metrics(test_batch_drift_signals, true_drift_bool)
        print('fpr:', fpr, 'acc:', acc)
        fprs.append(fpr)
        accs.append(acc)

    return fprs, accs