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


def prepare_data(df_X_ref, df_X_test, scaling, scaler, use_categorical, encoding=False, encoder=None):
    df_x_ref_num, df_x_ref_cat = divide_numeric_categorical(df_X_ref)
    df_x_test_num, df_x_test_cat = divide_numeric_categorical(df_X_test)

    if scaling:
        df_x_ref_num[df_x_ref_num.columns] = scaler.fit_transform(df_x_ref_num[df_x_ref_num.columns])
        df_x_test_num[df_x_test_num.columns] = scaler.fit_transform(df_x_test_num[df_x_test_num.columns])

    if use_categorical:
        if encoding:
            df_x_ref_cat[df_x_ref_cat.columns] = encoder.fit_transform(df_x_ref_cat[df_x_ref_cat.columns])
            df_x_test_cat[df_x_test_cat.columns] = encoder.fit_transform(df_x_test_cat[df_x_test_cat.columns])
        df_X_ref = df_x_ref_num.join(df_x_ref_cat)
        df_X_test = df_x_test_num.join(df_x_test_cat)
    else:
        df_X_ref = df_x_ref_num
        df_X_test = df_x_test_num

    return df_X_ref, df_X_test


def get_batches(df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches):
    X_ref_batches = np.array_split(df_X_ref, num_ref_batches)
    y_ref_batches = np.array_split(df_y_ref, num_ref_batches)
    X_test_batches = np.array_split(df_X_test, num_test_batches)
    y_test_batches = np.array_split(df_y_test, num_test_batches)
    return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches

