import pandas as pd
from scipy.io import arff

import preprocessing


def accept_data(file_path):
    """Accept an arff file and return its contents in a pandas dataframe"""
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    print('df', df)
    return df


def column_values_to_string(df, columns):
    for column in columns:
        df[column] = df[column].str.decode('utf-8')
    return df


def prepare_df_data(df):
    df_y = column_values_to_string(df[['class']], ['class'])
    df_x = df.drop(columns='class')
    df_x_numeric, df_x_categorical = preprocessing.divide_numeric_categorical(df_x)
    df_x_categorical = column_values_to_string(df_x_categorical, list(df_x_categorical.columns))
    df_x = df_x_numeric.join(df_x_categorical)
    return df_x, df_y


def get_clean_df(file_path):
    df = accept_data(file_path)
    df_x, df_y = prepare_df_data(df)

    return df_x, df_y

