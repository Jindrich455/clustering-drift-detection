from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

import accepting
import my_preprocessing
import ucdd
import ucdd_eval

if __name__ == '__main__':
    ucdd_eval.evaluate_ucdd(
        file_path='Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff',
        scaling="minmax",
        encoding="onehot",
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        random_state=2,
        additional_check=False
    )


    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    #     ('cat', OneHotEncoder(sparse=False), selector(dtype_exclude='number'))
    # ])
    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    #     ('cat', TargetEncoder(), selector(dtype_exclude='number'))
    # ])
    # transformer = FunctionTransformer(lambda x: x)
    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    # ])
