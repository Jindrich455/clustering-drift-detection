import csv

import pandas as pd
from matplotlib import pyplot as plt


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
trainbatches_path = ['trainbatches_always_all', 'trainbatches_never_all']
postfix = '.csv'

all_results = pd.read_csv(common_path + rel_paths[0] + postfix)

for rel_path in rel_paths[1:]:
    for trainbatch_path in trainbatches_path:
        res = pd.read_csv(common_path + rel_path + '_' + trainbatch_path + postfix)
        all_results = all_results.append(res)


all_results = all_results.drop(columns=['type_of_data', 'n_init', 'tol'])
all_results = all_results.sort_values(['dataset', 'encoding', 'train_batch_strategy', 'additional_check', 'width'])
print(all_results.to_string())
# all_results.to_csv(common_path + 'everything.csv', index=False)


sea_submajority = all_results.loc[(all_results['dataset'] == 'sea')
                                  & (all_results['train_batch_strategy'] == 'submajority')]
print(sea_submajority.to_string())
# sea_submajority.to_csv(common_path + 'sea_check_works.csv', index=False)


agraw1_meaningful = all_results.loc[(all_results['dataset'] == 'agraw1')
                                    & (all_results['train_batch_strategy'] == 'submajority')]

print('agraw1 meaningful')
print(agraw1_meaningful.to_string())

# colors = ["red", "green", "blue"]
# styles = ["-", "--", "-."]
# labels = ['encoding=exclude', 'encoding=onehot', 'encoding=target']
# agraw1_meaningful_with_check = agraw1_meaningful[agraw1_meaningful['additional_check'] == 'yes']
# for i in range(3):
#     agraw1_meaningful_one_result_set = agraw1_meaningful_with_check.iloc[(i * 6):((i + 1) * 6), :]
#     # line1, = plt.plot(agraw1_meaningful_one_result_set['width'], agraw1_meaningful_one_result_set['FPR_mean'],
#     #                   color=colors[i], linestyle=styles[i], marker='o',
#     #                   label=labels[i])
#     line1, = plt.plot(agraw1_meaningful_one_result_set['width'], agraw1_meaningful_one_result_set['latency_mean'],
#                       color=colors[i], linestyle=styles[i], marker='o',
#                       label=labels[i])
# plt.legend()
# plt.ylim((0, 1))
# plt.show()


# colors = ["red", "green", "blue"]
# styles = ["-", "--", "-."]
# labels = ['encoding=exclude', 'encoding=onehot', 'encoding=target']
# agraw1_meaningful_no_check = agraw1_meaningful[agraw1_meaningful['additional_check'] == 'no']
# for i in range(3):
#     agraw1_meaningful_one_result_set = agraw1_meaningful_no_check.iloc[(i * 6):((i + 1) * 6), :]
#     # line1, = plt.plot(agraw1_meaningful_one_result_set['width'], agraw1_meaningful_one_result_set['FPR_mean'],
#     #                   color=colors[i], linestyle=styles[i], marker='o',
#     #                   label=labels[i])
#     line1, = plt.plot(agraw1_meaningful_one_result_set['width'], agraw1_meaningful_one_result_set['latency_mean'],
#                       color=colors[i], linestyle=styles[i], marker='o',
#                       label=labels[i])
# plt.legend()
# plt.ylim((0, 1))
# plt.show()


# ag1 excl submaj yes

# ag1 1hot submaj no
# ag1 1hot submaj yes

# ag1 tgt submaj no
# ag1 tgt submaj yes

agraw2_meaningful = all_results.loc[(all_results['dataset'] == 'agraw2')
                                    & (all_results['train_batch_strategy'] == 'submajority')
                                    & (all_results['encoding'] == 'onehot')]

# print('agraw2 meaningful')
# print(agraw2_meaningful.to_string())
#
colors = ["red", "green"]
styles = ["-", "--"]
labels = ['no_additional_check', 'with_additional_check']
for i in range(2):
    agraw2_meaningful_one_result_set = agraw2_meaningful.iloc[(i * 6):((i + 1) * 6), :]
    # line1, = plt.plot(agraw2_meaningful_one_result_set['width'], agraw2_meaningful_one_result_set['FPR_mean'],
    #                   color=colors[i], linestyle=styles[i], marker='o',
    #                   label=labels[i])
    line1, = plt.plot(agraw2_meaningful_one_result_set['width'], agraw2_meaningful_one_result_set['latency_mean'],
                      color=colors[i], linestyle=styles[i], marker='o',
                      label=labels[i])
plt.legend()
plt.ylim((0, 1))
plt.show()


# ag2 1hot submaj no
# ag2 1hot submaj yes

sea_meaningful = all_results.loc[(all_results['dataset'] == 'sea')
                                 & (all_results['additional_check'] == 'yes')]

print('sea meaningful')
print(sea_meaningful.to_string())

# colors = ["red", "green", "blue"]
# styles = ["-", "--", "-."]
# labels = ['train_batch_strategy=all', 'train_batch_strategy=majority', 'train_batch_strategy=submajority']
# for i in range(3):
#     sea_meaningful_one_result_set = sea_meaningful.iloc[(i * 6):((i + 1) * 6), :]
#     # line1, = plt.plot(sea_meaningful_one_result_set['width'], sea_meaningful_one_result_set['FPR_mean'],
#     #                   color=colors[i], linestyle=styles[i], marker='o',
#     #                   label=labels[i])
#     line1, = plt.plot(sea_meaningful_one_result_set['width'], sea_meaningful_one_result_set['latency_mean'],
#                       color=colors[i], linestyle=styles[i], marker='o',
#                       label=labels[i])
# plt.legend()
# plt.ylim((0, 1))
# plt.show()

# sea excl all yes
# sea excl maj yes
# sea excl submaj yes

# agraw1_meaningful.to_csv(common_path + 'agraw1_meaningful.csv', index=False)
# agraw2_meaningful.to_csv(common_path + 'agraw2_meaningful.csv', index=False)
# sea_meaningful.to_csv(common_path + 'sea_meaningful.csv', index=False)
