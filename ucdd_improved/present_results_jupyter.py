import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from core import ucdd


# print(os.listdir('results_after_analysis_jupyter'))
# path_to_results = 'results_after_analysis_jupyter'
# for filename in os.listdir(path_to_results):
#     df = pd.read_csv(path_to_results + '/' + filename)
#     name = '_'.join([df['dataset'].iloc[0], df['encoding'].iloc[0], 'minref' + str(df['min_ref_batches_drift'].iloc[0]),
#            ('yescheck' if df['additional_check'].iloc[0] == 'yes' else 'nocheck')])
#     print(name)
#
#     clearer_df = df[["width", "FPR_mean", "latency_mean"]]
#     clearer_df['FPR_mean'] = clearer_df['FPR_mean'].apply(lambda x: round(x, 2))
#     clearer_df['latency_mean'] = clearer_df['latency_mean'].apply(lambda x: round(x, 2))
#     print(clearer_df)
#     clearer_df.to_csv('jupyter_presentable_results/' + name + '.csv', index=False)

ucdd.all_drifting_batches_parallel(
    [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],
    [np.array([7, 8]), np.array([9, 10])],
    min_ref_batches_drift=0.3,
    additional_check=True,
    n_init=100,
    max_iter=77000,
    tol=0,
    random_state=0
)
