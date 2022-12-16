import itertools

import pandas as pd
import numpy as np
import sklearn

from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from mssw import mssw_supported_parameters as spms

import accepting
import mssw.mssw_eval


def eval_one_parameter_set(
        data_path,
        encoding,
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

    ref_batches = np.array_split(reference_data, num_ref_batches)
    test_batches = np.array_split(testing_data, num_test_batches)

    return mssw.mssw_eval.all_drifting_batches_randomness_robust(ref_batches, test_batches)


def eval_multiple_parameter_sets(
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
    arg_tuples = list(itertools.product(data_paths, encodings))
    argument_results = []
    for i, arg_tuple in enumerate(arg_tuples):
        print('argument combination #', i)
        data_path = arg_tuple[0]
        encoding = arg_tuple[1]
        print('data path')
        print(data_path)
        print('encoding')
        print(encoding)
        runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se = eval_one_parameter_set(
            data_path,
            encoding,
            test_fraction,
            num_ref_batches,
            num_test_batches,
            true_drift_idx,
            num_clusters=num_clusters,
            first_random_state=first_random_state,
            coeff=coeff,
            min_runs=min_runs,
            std_err_threshold=std_err_threshold
        )
        argument_results.append((data_path, encoding, runs_results_bool, fpr_mean, fpr_se, latency_mean, latency_se))

    return argument_results