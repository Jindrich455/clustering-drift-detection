import itertools

import pandas as pd
import numpy as np
import sklearn

from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

import ucdd.ucdd_eval
from ucdd import ucdd_supported_parameters as spms

from mssw import accepting


def eval_one_parameter_set(data_path, encoding, test_fraction, num_ref_batches, num_test_batches, true_drift_idx,
                           train_batch_strategy, additional_check,
                           n_init=10, max_iter=300, tol=1e-4, first_random_state=0,
                           min_runs=10, std_err_threshold=0.05):
    df_x, df_y = accepting.get_clean_df(data_path)

    df_y = pd.DataFrame(LabelEncoder().fit_transform(df_y))
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=test_fraction, shuffle=False)

    df_x_ref_num, df_x_ref_cat = accepting.divide_numeric_categorical(df_x_ref)
    df_x_test_num, df_x_test_cat = accepting.divide_numeric_categorical(df_x_test)

    if df_x_ref_cat.shape[0] == 0:
        encoding = spms.Encoders.EXCLUDE

    if encoding == spms.Encoders.EXCLUDE:
        df_x_ref_cat_transformed = pd.DataFrame(None)
        df_x_test_cat_transformed = pd.DataFrame(None)
    elif encoding == spms.Encoders.ONEHOT:
        ref_index = df_x_ref_cat.index
        test_index = df_x_test_cat.index
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(df_x_ref_cat)
        df_x_ref_cat_transformed = pd.DataFrame(encoder.transform(df_x_ref_cat))
        df_x_test_cat_transformed = pd.DataFrame(encoder.transform(df_x_test_cat))
        df_x_ref_cat_transformed.set_index(ref_index, inplace=True)
        df_x_test_cat_transformed.set_index(test_index, inplace=True)
    elif encoding == spms.Encoders.TARGET:
        ref_index = df_x_ref_cat.index
        test_index = df_x_test_cat.index
        encoder = TargetEncoder()
        encoder.fit(df_x_ref_cat, df_y_ref)
        df_x_ref_cat_transformed = pd.DataFrame(encoder.transform(df_x_ref_cat))
        df_x_test_cat_transformed = pd.DataFrame(encoder.transform(df_x_test_cat))
        df_x_ref_cat_transformed.set_index(ref_index, inplace=True)
        df_x_test_cat_transformed.set_index(test_index, inplace=True)
    else:
        raise NameError('The encoding', encoding, 'is not supported')

    reference_data = df_x_ref_num.join(df_x_ref_cat_transformed, lsuffix='_num').to_numpy()
    testing_data = df_x_test_num.join(df_x_test_cat_transformed, lsuffix='_num').to_numpy()

    scaler = MinMaxScaler()
    scaler.fit(reference_data)
    reference_data = scaler.transform(reference_data)
    testing_data = scaler.transform(testing_data)

    ref_batches = np.array_split(reference_data, num_ref_batches)
    test_batches = np.array_split(testing_data, num_test_batches)

    return ucdd.ucdd_eval.all_drifting_batches_randomness_robust(
        ref_batches,
        test_batches,
        train_batch_strategy=train_batch_strategy,
        additional_check=additional_check,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        true_drift_idx=true_drift_idx,
        first_random_state=first_random_state,
        min_runs=min_runs,
        std_err_threshold=std_err_threshold
    )


def eval_multiple_parameter_sets(data_paths, encodings, test_fraction, num_ref_batches, num_test_batches,
                                 true_drift_idx,
                                 train_batch_strategies, additional_checks,
                                 n_inits=[10], max_iters=[300], tols=[1e-4], first_random_state=0,
                                 min_runs=10, std_err_threshold=0.05):
    arg_tuples = list(itertools.product(
        data_paths, encodings, train_batch_strategies, additional_checks, n_inits, max_iters, tols))
    argument_results = []
    for i, arg_tuple in enumerate(arg_tuples):
        print('argument combination #', i, 'of', len(arg_tuples))
        data_path = arg_tuple[0]
        encoding = arg_tuple[1]
        train_batch_strategy = arg_tuple[2]
        additional_check = arg_tuple[3]
        n_init = arg_tuple[4]
        max_iter = arg_tuple[5]
        tol = arg_tuple[6]
        print('data path')
        print(data_path)
        print('encoding')
        print(encoding)
        print('train batch strategy')
        print(train_batch_strategy)
        print('additional check')
        print(additional_check)
        print('n_init')
        print(n_init)
        print('max_iter')
        print(max_iter)
        print('tol')
        print(tol)
        runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se = eval_one_parameter_set(
            data_path,
            encoding,
            test_fraction,
            num_ref_batches,
            num_test_batches,
            true_drift_idx,
            train_batch_strategy=train_batch_strategy,
            additional_check=additional_check,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            first_random_state=first_random_state,
            min_runs=min_runs,
            std_err_threshold=std_err_threshold
        )
        results_dict = {
            'data_path': data_path,
            'encoding': encoding.name.lower(),
            'train_batch_strategy': train_batch_strategy.name.lower(),
            'additional_check': 'yes' if additional_check else 'no',
            'n_init': n_init,
            'max_iter': max_iter,
            'tol': tol,
            'first_random_state': first_random_state,
            'min_runs': min_runs,
            'std_err_threshold': std_err_threshold,
            'runs_results_bool': runs_results_bool,
            'fpr_mean': fpr_mean,
            'fpr_se': fpr_se,
            'latency_mean': latency_mean,
            'latency_se': latency_se
        }
        argument_results.append(results_dict)

    return argument_results
