import csv

import pandas as pd
from matplotlib import pyplot as plt


def plot_dataset(dataset):
    colors = ["red", "green"]
    styles = ["-", "--"]
    labels = ['mean_FPR', 'mean_latency']
    # sea_results = all_results[all_results['dataset'] == 'sea']
    plt.plot(dataset['width'], dataset['FPR_mean'],
             color=colors[0], linestyle=styles[0], marker='o',
             label=labels[0]
             )
    plt.plot(dataset['width'], dataset['latency_mean'],
             color=colors[1], linestyle=styles[1], marker='o',
             label=labels[1]
             )

    plt.legend()
    plt.ylim((0, 1))
    plt.show()


common_path = 'results_after_analysis/'
rel_paths = [
    'sea',
    'agraw1_exclude',
    'agraw1_onehot',
    'agraw1_target',
    'agraw2_exclude',
    'agraw2_onehot',
    'agraw2_target'
]
postfix = '.csv'

all_results = pd.read_csv(common_path + rel_paths[0] + postfix)

for rel_path in rel_paths[1:]:
    res = pd.read_csv(common_path + rel_path + postfix)
    all_results = all_results.append(res)


print(all_results.to_string())
all_results = all_results.drop(columns=['type_of_data', 'n_init', 'tol'])
all_results.to_csv(common_path + 'everything.csv', index=False)


# all_agraw = res_dict['agraw1_exclude'].join(res_dict['agraw1_onehot'].join(res_dict['agraw1_target']))
# print('all agraw', all_agraw)

# sea_results = all_results[all_results['dataset'] == 'sea']
# plot_dataset(sea_results)

exclude_mask = all_results['encoding'] == 'exclude'
onehot_mask = all_results['encoding'] == 'onehot'
target_mask = all_results['encoding'] == 'target'

agraw1_mask = all_results['dataset'] == 'agraw1'
# agraw1_exclude_results = all_results[agraw1_mask & exclude_mask]
# plot_dataset(agraw1_exclude_results)
# agraw1_onehot_results = all_results[agraw1_mask & onehot_mask]
# plot_dataset(agraw1_onehot_results)
# agraw1_target_results = all_results[agraw1_mask & target_mask]
# plot_dataset(agraw1_target_results)

agraw2_mask = all_results['dataset'] == 'agraw2'
# agraw2_exclude_results = all_results[agraw2_mask & exclude_mask]
# plot_dataset(agraw2_exclude_results)
# agraw2_onehot_results = all_results[agraw2_mask & onehot_mask]
# plot_dataset(agraw2_onehot_results)
# agraw2_target_results = all_results[agraw2_mask & target_mask]
# plot_dataset(agraw2_target_results)

