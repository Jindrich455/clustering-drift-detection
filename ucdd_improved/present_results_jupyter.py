import pandas as pd
from matplotlib import pyplot as plt
import os


print(os.listdir('results_after_analysis_jupyter'))
path_to_results = 'results_after_analysis_jupyter'
for filename in os.listdir(path_to_results):
    df = pd.read_csv(path_to_results + '/' + filename)
    name = '_'.join([df['dataset'].iloc[0], df['encoding'].iloc[0], df['train_batch_strategy'].iloc[0],
           ('yescheck' if df['additional_check'].iloc[0] == 'yes' else 'nocheck')])
    print(name)

    clearer_df = df[["width", "FPR_mean", "latency_mean"]]
    clearer_df['FPR_mean'] = clearer_df['FPR_mean'].apply(lambda x: round(x, 2))
    clearer_df['latency_mean'] = clearer_df['latency_mean'].apply(lambda x: round(x, 2))
    print(clearer_df)
    clearer_df.to_csv('jupyter_presentable_results/' + name + '.csv', index=False)
