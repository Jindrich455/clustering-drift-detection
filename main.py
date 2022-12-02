from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

import accepting
import my_preprocessing
import ucdd
import ucdd_eval
import ucdd_visual_inspection
import supported_parameters as spms

if __name__ == '__main__':
    ucdd_eval.evaluate_ucdd(
        file_path='Datasets_concept_drift/synthetic_data/abrupt_drift/sea_1_abrupt_drift_0_noise_balanced.arff',
        scaling=spms.Scalers.MINMAX,
        encoding=spms.Encoders.EXCLUDE,
        test_size=0.7,
        num_ref_batches=3,
        num_test_batches=7,
        random_state=2,
        additional_check=True,
        detect_all_training_batches=False,
        use_pyclustering=True,
        metric_id=spms.Distances.EUCLIDEAN
    )



    # ucdd_eval.evaluate_ucdd(
    #     file_path='tests/test_datasets/drift_2d.arff',
    #     scaling="minmax",
    #     encoding="none",
    #     test_size=0.5,
    #     num_ref_batches=1,
    #     num_test_batches=1,
    #     random_state=0,
    #     additional_check=True,
    #     use_pyclustering=True,
    #     debug=True
    # )

    # ucdd_visual_inspection.show_ucdd()


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
