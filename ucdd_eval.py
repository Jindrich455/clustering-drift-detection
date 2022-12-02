import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import accepting
import my_preprocessing
import ucdd
import ucdd_pyclustering
import supported_parameters as spms

accepted_scalers = ["minmax"]
accepted_encoders = ["onehot", "ordinal", "target"]
accepted_distance_measures = [""]


# def scale_with(df_x_num, scaler_id):
#     if scaler_id == spms.Scalers.MINMAX:
#         scaler = MinMaxScaler()
#         df_x_num = pd.DataFrame(scaler.fit_transform(df_x_num))
#     # else assume no scaling should be done
#     return df_x_num


def scale_with(df_x_ref_num_new, df_x_test_num_new, scaler_id):
    if scaler_id == spms.Scalers.MINMAX:
        scaler = MinMaxScaler()
        scaler.fit(df_x_ref_num_new)
        df_x_ref = pd.DataFrame(scaler.transform(df_x_ref_num_new))
        df_x_test = pd.DataFrame(scaler.transform(df_x_test_num_new))
    elif scaler_id == spms.Scalers.ID:
        df_x_ref = df_x_ref_num_new
        df_x_test = df_x_test_num_new
    else:
        raise NameError('Unsupported scaling!!')
    return df_x_ref, df_x_test


# def encode_with(df_x_cat, df_y, encoder_id):
#     if encoder_id == spms.Encoders.ONEHOT:
#         encoder = OneHotEncoder(sparse=False)
#         df_x_cat = pd.DataFrame(encoder.fit_transform(df_x_cat))
#     elif encoder_id == spms.Encoders.ORDINAL:
#         encoder = OrdinalEncoder()
#         df_x_cat = pd.DataFrame(encoder.fit_transform(df_x_cat))
#     elif encoder_id == spms.Encoders.TARGET:
#         encoder = TargetEncoder(df_x_cat)
#         df_x_cat = encoder.fit_transform(df_x_cat, df_y)
#     elif encoder_id == spms.Encoders.EXCLUDE:
#         # exclude categorical data from drift detection
#         df_x_cat = pd.DataFrame(None)
#     # else assume no encoding should be done
#     return df_x_cat


def transform_with_fitted_encoder(df_x_ref_cat, df_x_test_cat, encoder):
    # Important: after encoder's transform, the dataframe indices must still match
    ref_index = df_x_ref_cat.index
    test_index = df_x_test_cat.index
    df_x_ref_cat_transformed = pd.DataFrame(encoder.transform(df_x_ref_cat))
    df_x_test_cat_transformed = pd.DataFrame(encoder.transform(df_x_test_cat))
    df_x_ref_cat_transformed.set_index(ref_index, inplace=True)
    df_x_test_cat_transformed.set_index(test_index, inplace=True)
    return df_x_ref_cat_transformed, df_x_test_cat_transformed


def encode_with(df_x_ref_cat, df_x_test_cat, df_y_ref, encoder_id):
    # otherwise fit an encoder and transform the data accordingly
    if encoder_id == spms.Encoders.EXCLUDE:
        # simplest behaviour: categorical data are simply excluded
        df_x_ref_cat_transformed = pd.DataFrame(None)
        df_x_test_cat_transformed = pd.DataFrame(None)
    else:
        if encoder_id == spms.Encoders.ONEHOT:
            encoder = OneHotEncoder(sparse=False)
            encoder.fit(df_x_ref_cat)
        elif encoder_id == spms.Encoders.TARGET:
            encoder = TargetEncoder()
            encoder.fit(df_x_ref_cat, df_y_ref)
        elif encoder_id == spms.Encoders.ORDINAL:
            encoder = OrdinalEncoder()
            encoder.fit(df_x_ref_cat)
        else:
            raise NameError('Unsupported encoding!!')
        df_x_ref_cat_transformed, df_x_test_cat_transformed = transform_with_fitted_encoder(
            df_x_ref_cat, df_x_test_cat, encoder)

    return df_x_ref_cat_transformed, df_x_test_cat_transformed


def get_batches(df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches):
    """Divide reference and testing data and labels into lists of batches"""
    X_ref_batches = np.array_split(df_X_ref, num_ref_batches)
    y_ref_batches = np.array_split(df_y_ref, num_ref_batches)
    X_test_batches = np.array_split(df_X_test, num_test_batches)
    y_test_batches = np.array_split(df_y_test, num_test_batches)
    return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches


def preprocess_df_x(df_x_ref, df_x_test, df_y_ref, scaling, encoding):
    # Categorical data is always either excluded or converted to numerical!!
    df_x_ref_num, df_x_ref_cat = accepting.divide_numeric_categorical(df_x_ref)
    df_x_test_num, df_x_test_cat = accepting.divide_numeric_categorical(df_x_test)

    df_x_ref_cat_encoded, df_x_test_cat_encoded = encode_with(
        df_x_ref_cat, df_x_test_cat, df_y_ref, encoding)

    df_x_ref_num_new = df_x_ref_num.join(df_x_ref_cat_encoded, lsuffix='_num')
    df_x_test_num_new = df_x_test_num.join(df_x_test_cat_encoded, lsuffix='_num')

    df_x_ref_final, df_x_test_final = scale_with(df_x_ref_num_new, df_x_test_num_new, scaling)

    df_x_ref_final.columns = df_x_ref_final.columns.astype(str)
    df_x_test_final.columns = df_x_test_final.columns.astype(str)

    # df_x_num = scale_with(df_x_num, scaling)
    # print('df_x_num')
    # print(df_x_num)
    # df_x_cat = encode_with(df_x_cat, df_y, encoding)
    # print('df_x_cat')
    # print(df_x_cat)
    # df_x = df_x_num.join(df_x_cat, lsuffix='_num')
    # print('final df_x')
    # print(df_x)

    return df_x_ref_final, df_x_test_final


def evaluate_ucdd(file_path, scaling, encoding, test_size, num_ref_batches, num_test_batches,
                  random_state, additional_check, detect_all_training_batches,
                  debug=False, use_pyclustering=False, metric_id=spms.Distances.EUCLIDEAN):
    df_x, df_y = accepting.get_clean_df(file_path)

    # convert labels to 0 and 1
    df_y = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(df_y))

    # split data to training and testing (with a joint dataframe)
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=test_size, shuffle=False)

    # do all the necessary data transformations (e.g. one-hot encoding, scaling)
    # --> might be different for each dataset
    df_x_ref, df_x_test = preprocess_df_x(df_x_ref, df_x_test, df_y_ref, scaling=scaling, encoding=encoding)

    # divide the data in batches
    x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = get_batches(
        df_x_ref, df_x_test, df_y_ref, df_y_test, num_ref_batches=num_ref_batches, num_test_batches=num_test_batches
    )

    # use ucdd on the batched data and find drift locations
    drift_locations = []
    if use_pyclustering:
        drift_locations = ucdd_pyclustering.drift_occurrences_list(
            x_ref_batches, x_test_batches, random_state=random_state, additional_check=additional_check,
            detect_all_training_batches=detect_all_training_batches,
            debug=debug,
            metric_id=metric_id
        )
    else:
        drift_locations = ucdd.drift_occurrences_list(
            x_ref_batches, x_test_batches, random_state=random_state, additional_check=additional_check,
            detect_all_training_batches=detect_all_training_batches,
            debug=debug)
    print('drift locations', drift_locations)
    return drift_locations
