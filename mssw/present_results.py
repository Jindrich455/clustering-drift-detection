import csv

import pandas as pd


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
