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

accepted_scalers = ["minmax"]
accepted_encoders = ["onehot", "ordinal", "target"]


def scale_with(df_x_num, scaler_name):
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
        df_x_num = pd.DataFrame(scaler.fit_transform(df_x_num))
    # else assume no scaling should be done
    return df_x_num


def encode_with(df_x_cat, df_y, encoder_name):
    if encoder_name == "onehot":
        encoder = OneHotEncoder(sparse=False)
        df_x_cat = pd.DataFrame(encoder.fit_transform(df_x_cat))
    elif encoder_name == "ordinal":
        encoder = OrdinalEncoder()
        df_x_cat = pd.DataFrame(encoder.fit_transform(df_x_cat))
    elif encoder_name == "target":
        encoder = TargetEncoder(df_x_cat)
        df_x_cat = encoder.fit_transform(df_x_cat, df_y)
    # else assume no encoding should be done
    return df_x_cat


def get_batches(df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches):
    """Divide reference and testing data and labels into lists of batches"""
    X_ref_batches = np.array_split(df_X_ref, num_ref_batches)
    y_ref_batches = np.array_split(df_y_ref, num_ref_batches)
    X_test_batches = np.array_split(df_X_test, num_test_batches)
    y_test_batches = np.array_split(df_y_test, num_test_batches)
    return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches


def preprocess_df_x(df_x_num, df_x_cat, df_y, scaling, encoding):
    df_x_num = scale_with(df_x_num, scaling)
    print('df_x_num')
    print(df_x_num)
    df_x_cat = encode_with(df_x_cat, df_y, encoding)
    print('df_x_cat')
    print(df_x_cat)
    df_x = df_x_num.join(df_x_cat, lsuffix='_num')
    df_x.columns = df_x.columns.astype(str)
    print('final df_x')
    print(df_x)
    return df_x


def evaluate_ucdd():
    df_x_num, df_x_cat, df_y = accepting.get_clean_df(
            'Datasets_concept_drift/synthetic_data/gradual_drift/agraw1_1_gradual_drift_0_noise_balanced_5.arff')

    # do all the necessary data transformations (e.g. scaling, one-hot encoding)
    # --> might be different for each dataset
    df_y = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(df_y))
    df_x = preprocess_df_x(df_x_num, df_x_cat, df_y, scaling="minmax", encoding="ordinal")

    # split data to training and testing (with a joint dataframe)
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=0.7, shuffle=False)

    # divide the data in batches
    x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = get_batches(
        df_x_ref, df_x_test, df_y_ref, df_y_test, num_ref_batches=3, num_test_batches=7
    )

    # use ucdd on the batched data and find drift locations
    drift_locations = ucdd.drift_occurrences_list(x_ref_batches, x_test_batches, random_state=2)
    print('drift locations', drift_locations)
    pass
