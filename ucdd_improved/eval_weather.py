from sklearn.preprocessing import MinMaxScaler
from eval_helpers import helpers
import pandas as pd
import csv
import time
from eval_helpers import ucdd_eval_real_world


df = pd.read_csv("../Datasets_concept_drift/real_world_data/weather_dataset.csv")

X = df.drop(columns=['Unnamed: 0', 'Label_Rain'])
y = df["Label_Rain"]

X_ref = X[:6053]
X_test = X[6053:]
y_ref = y[:6053]
y_test = y[6053:]

X_ref = X_ref.to_numpy()
X_test = X_test.to_numpy()
y_ref = y_ref.to_numpy()
y_test = y_test.to_numpy()

scaler = MinMaxScaler()
scaler.fit(X_ref)
X_ref = scaler.transform(X_ref)
X_test = scaler.transform(X_test)


X_test_batches_year, y_test_batches_year = helpers.split_to_fixed_size_batches(X_test, y_test, batch_size=365)
X_ref_batches_year, y_ref_batches_year = helpers.split_to_fixed_size_batches(X_ref, y_ref, batch_size=365)

X_test_batches_month, y_test_batches_month = helpers.split_to_fixed_size_batches(X_test, y_test, batch_size=30)
X_ref_batches_month, y_ref_batches_month = helpers.split_to_fixed_size_batches(X_ref, y_ref, batch_size=30)


true_drift_bool_month = []
with open('../Datasets_concept_drift/real_world_data_drifts/weather/weather_monthly_drifts.csv') as f:
    rdr = csv.reader(f, delimiter=',')
    for row in rdr:
        true_drift_bool_month.append(row)
    true_drift_bool_month = true_drift_bool_month[0] # only one row of important data
    true_drift_bool_month = [b == 'True' for b in true_drift_bool_month]
print(true_drift_bool_month)

print('monthly drifts found')

start_time = time.time()
_, fpr_mean_month, _, det_acc_mean_month, _ = ucdd_eval_real_world.all_drifting_batches_randomness_robust(
    reference_data_batches=X_ref_batches_month,
    testing_data_batches=X_test_batches_month,
    true_drift_bool=true_drift_bool_month,
    min_ref_batches_drift=0.3,
    additional_check=True,
    n_init=100, max_iter=18000, tol=0,
    first_random_state=0,
    min_runs=2, std_err_threshold=0.05
)
end_time = time.time()
print('time taken:', end_time - start_time)
print('monthly mean FPR:', fpr_mean_month)
print('monthly mean detection accuracy:', det_acc_mean_month)
