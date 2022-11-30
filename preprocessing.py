import accepting
import sklearn.model_selection
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def divide_numeric_categorical(df_x):
    df_x_numeric = df_x.select_dtypes(include=[np.number])
    df_x_categorical = df_x.select_dtypes(exclude=[np.number])
    return df_x_numeric, df_x_categorical


# def prepare_data(df_X_ref, df_X_test, scaling, scaler, use_categorical, encoding=False, encoder=None):
#     df_x_ref_num, df_x_ref_cat = divide_numeric_categorical(df_X_ref)
#     df_x_test_num, df_x_test_cat = divide_numeric_categorical(df_X_test)
#
#     if scaling:
#         df_x_ref_num[df_x_ref_num.columns] = scaler.fit_transform(df_x_ref_num[df_x_ref_num.columns])
#         df_x_test_num[df_x_test_num.columns] = scaler.fit_transform(df_x_test_num[df_x_test_num.columns])
#
#     if use_categorical:
#         if encoding:
#             df_x_ref_cat[df_x_ref_cat.columns] = encoder.fit_transform(df_x_ref_cat[df_x_ref_cat.columns])
#             df_x_test_cat[df_x_test_cat.columns] = encoder.fit_transform(df_x_test_cat[df_x_test_cat.columns])
#         df_X_ref = df_x_ref_num.join(df_x_ref_cat)
#         df_X_test = df_x_test_num.join(df_x_test_cat)
#     else:
#         df_X_ref = df_x_ref_num
#         df_X_test = df_x_test_num
#
#     return df_X_ref, df_X_test


def transform_df_with(df, transformer):
    df[df.columns] = transformer.fit_transform(df[df.columns])
    return df


def prepare_data(df_x, scaling, scaler, use_categorical, encoding=False, encoder=None):
    df_x_num, df_x_cat = divide_numeric_categorical(df_x)
    if scaling:
        transform_df_with(df_x_num, scaler)

    if use_categorical:
        if encoding:
            transform_df_with(df_x_cat, encoder)
        df_x = df_x_num.join(df_x_cat)
    else:
        df_x = df_x_num

    return df_x


def prepare_data_and_get_batches(df_x, df_y, test_fraction, num_ref_batches, num_test_batches,
                                 scaling, scaler, use_categorical, encoding=False, encoder=None):
    df_x = prepare_data(df_x, scaling, scaler, use_categorical, encoding=False, encoder=None)

    df_X_ref, df_X_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=test_fraction, shuffle=False)

    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = get_batches(
        df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches
    )
    return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches


def get_batches(df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches):
    X_ref_batches = np.array_split(df_X_ref, num_ref_batches)
    y_ref_batches = np.array_split(df_y_ref, num_ref_batches)
    X_test_batches = np.array_split(df_X_test, num_test_batches)
    y_test_batches = np.array_split(df_y_test, num_test_batches)
    return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches

