print('plain prints work')


from sklearn.preprocessing import MinMaxScaler
from eval_helpers import helpers, accepting
import pandas as pd
import csv
import time
from eval_helpers import ucdd_eval_real_world
from core import ucdd_eval
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn
import numpy as np


print('imports work')


agraw2_onehot_reference_batches = {}
agraw2_onehot_testing_batches = {}

agraw2_abrupt_path = "../Datasets_concept_drift/synthetic_data/abrupt_drift/agraw2_1_abrupt_drift_0_noise_balanced.arff"

df_x, df_y = accepting.get_clean_df(agraw2_abrupt_path)
df_y = pd.DataFrame(LabelEncoder().fit_transform(df_y))

print('accepting a file works')

df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
    df_x, df_y, test_size=0.7, shuffle=False)

print('splitting to train test works')

df_x_ref_num, df_x_ref_cat = accepting.divide_numeric_categorical(df_x_ref)
df_x_test_num, df_x_test_cat = accepting.divide_numeric_categorical(df_x_test)

print('dividing numeric categorical works')

ref_index = df_x_ref_cat.index
test_index = df_x_test_cat.index
encoder = OneHotEncoder(sparse=False)
encoder.fit(df_x_ref_cat)
df_x_ref_cat_transformed = pd.DataFrame(encoder.transform(df_x_ref_cat))
df_x_test_cat_transformed = pd.DataFrame(encoder.transform(df_x_test_cat))
df_x_ref_cat_transformed.set_index(ref_index, inplace=True)
df_x_test_cat_transformed.set_index(test_index, inplace=True)

print('onehot encoding works')

reference_data = df_x_ref_num.join(df_x_ref_cat_transformed, lsuffix='_num').to_numpy()
testing_data = df_x_test_num.join(df_x_test_cat_transformed, lsuffix='_num').to_numpy()
scaler = MinMaxScaler()
scaler.fit(reference_data)
reference_data = scaler.transform(reference_data)
testing_data = scaler.transform(testing_data)

print('scaling data works')

num_ref_batches = 3
num_test_batches = 7
ref_batches = np.array_split(reference_data, num_ref_batches)
test_batches = np.array_split(testing_data, num_test_batches)

agraw2_onehot_reference_batches[agraw2_abrupt_path] = ref_batches
agraw2_onehot_testing_batches[agraw2_abrupt_path] = test_batches

print('receiving batches works')

agraw2_onehot_stats1 = {}

print('before ucdd eval')

start_time = time.time()

runs_results_bool, final_fpr_mean, fpr_std_err, final_latency_mean, latency_std_err = \
    ucdd_eval.all_drifting_batches_randomness_robust(
    agraw2_onehot_reference_batches[agraw2_abrupt_path],
    agraw2_onehot_testing_batches[agraw2_abrupt_path],
    min_ref_batches_drift=0.3,
    additional_check=True,
    n_init=100,
    max_iter=37000,
    tol=0,
    true_drift_idx=2,
    min_runs=2,
    parallel=False
)

print('after ucdd eval')

agraw2_onehot_stats1[agraw2_abrupt_path] = {
    'runs_results_bool': runs_results_bool,
    'final_fpr_mean': final_fpr_mean,
    'fpr_std_err': fpr_std_err,
    'final_latency_mean': final_latency_mean,
    'latency_std_err': latency_std_err
}

print('agraw2 STATS')
print(agraw2_onehot_stats1)

final_result_dict = {
    'type_of_data': [], 'dataset': [], 'drift': [], 'width': [], 'encoding': [],
    'min_ref_batches_drift': [], 'additional_check': [],
    'n_init': [], 'max_iter': [], 'tol': [],
    'FPR_mean': [], 'latency_mean': []
}

for data_path, stats_dict in agraw2_onehot_stats1.items():
    synthetic_filename_info = helpers.synthetic_data_information(data_path)
    encoding = 'onehot'
    fpr_mean = float(stats_dict['final_fpr_mean'])
    latency_mean = float(stats_dict['final_latency_mean'])

    final_result_dict['type_of_data'].append(synthetic_filename_info['type_of_data'])
    final_result_dict['dataset'].append(synthetic_filename_info['dataset_name'])
    final_result_dict['drift'].append(synthetic_filename_info['drift_type'])
    final_result_dict['width'].append(synthetic_filename_info['drift_width'])
    final_result_dict['encoding'].append(encoding)
    final_result_dict['min_ref_batches_drift'].append(0.3)
    final_result_dict['additional_check'].append('yes')
    final_result_dict['n_init'].append(100)
    final_result_dict['max_iter'].append(37000)
    final_result_dict['tol'].append(0)
    final_result_dict['FPR_mean'].append(fpr_mean)
    final_result_dict['latency_mean'].append(latency_mean)

final_result_df = pd.DataFrame.from_dict(final_result_dict)
sorted_final_result_df = final_result_df.sort_values(['drift', 'dataset', 'encoding', 'width'])
final_result_df.to_csv('agraw2_onehot_jupyter_results.csv', index=False)

print('time taken:', time.time() - start_time)
