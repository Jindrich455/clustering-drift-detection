import pandas as pd
import sklearn
import supported_parameters as spms
from matplotlib import pyplot as plt
from pandas import Series
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

import accepting
from sklearn.compose import make_column_selector as selector

import my_preprocessing
import ucdd
import ucdd_eval
import ucdd_pyclustering


def show_ucdd(
        file_path='tests/test_datasets/drift_2d.arff',
        scaling=spms.Scalers.MINMAX,
        encoding=spms.Encoders.EXCLUDE,
        test_size=0.5,
        num_ref_batches=1,
        num_test_batches=1,
        random_state=0,
        additional_check=False):
    # for some reason, the values must be default arguments; not regular variables

    # df_x_num, df_x_cat, df_y = accepting.get_clean_df(file_path)
    #
    # # do all the necessary data transformations (e.g. scaling, one-hot encoding)
    # # --> might be different for each dataset
    # df_y = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(df_y))
    # df_x = ucdd_eval.preprocess_df_x(df_x_num, df_x_cat, df_y, scaling=scaling, encoding=encoding)
    #
    # # split data to training and testing (with a joint dataframe)
    # df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
    #     df_x, df_y, test_size=test_size, shuffle=False)
    #
    # # divide the data in batches
    # x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = ucdd_eval.get_batches(
    #     df_x_ref, df_x_test, df_y_ref, df_y_test, num_ref_batches=num_ref_batches, num_test_batches=num_test_batches
    # )

    df_x, df_y = accepting.get_clean_df(file_path)

    # convert labels to 0 and 1
    df_y = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(df_y))

    # split data to training and testing (with a joint dataframe)
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=test_size, shuffle=False)

    # do all the necessary data transformations (e.g. one-hot encoding, scaling)
    # --> might be different for each dataset
    df_x_ref, df_x_test = ucdd_eval.preprocess_df_x(df_x_ref, df_x_test, df_y_ref, scaling=scaling, encoding=encoding)
    # reindex data to make sure the indices match
    df_y_ref.set_index(df_x_ref.index, inplace=True)
    df_y_test.set_index(df_x_test.index, inplace=True)

    # divide the data in batches
    x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = ucdd_eval.get_batches(
        df_x_ref, df_x_test, df_y_ref, df_y_test, num_ref_batches=num_ref_batches, num_test_batches=num_test_batches
    )

    show_initial_data(x_ref_batches, y_ref_batches, x_test_batches, y_test_batches)

    # use ucdd on the batched data and find drift locations
    drift_locations = ucdd_pyclustering.drift_occurrences_list(
        x_ref_batches, x_test_batches, random_state=random_state, additional_check=additional_check, show_2d_plots=True,
        detect_all_training_batches=False,
        metric_id=spms.Distances.EUCLIDEAN
    )

    # df_x, df_y = accepting.get_clean_df('tests/test_datasets/drift_2d.arff')
    #
    # transformer = ColumnTransformer([
    #     ('num', MinMaxScaler(), selector(dtype_include='number')),
    # ])
    #
    # X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = my_preprocessing.transform_data_and_get_batches(
    #     df_x, df_y, test_fraction=0.5, num_ref_batches=1, num_test_batches=1, transformer=transformer
    # )
    #
    #
    # drift_occurrences = ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state=0, show_2d_plots=True)


def divide_to_positive_negative(df_X, df_y):
    print('df_X')
    print(df_X)
    print('df_y')
    print(df_y)
    labels_list = Series(df_y.iloc[:, 0]).astype(bool)
    pos = df_X[labels_list]
    neg = df_X[~labels_list]
    return pos, neg


def show_initial_data(X_ref_batches, y_ref_batches, X_test_batches, y_test_batches):
    df_X_ref = X_ref_batches[0]
    df_y_ref = y_ref_batches[0]
    df_X_test = X_test_batches[0]
    df_y_test = y_test_batches[0]

    c_ref = 'g'
    positive_ref, negative_ref = divide_to_positive_negative(df_X_ref, df_y_ref)
    print('positive_ref')
    print(positive_ref)
    print('negative_ref')
    print(negative_ref)
    plt.scatter(positive_ref.iloc[:, 0], positive_ref.iloc[:, 1], marker=',', c=c_ref)
    plt.scatter(negative_ref.iloc[:, 0], negative_ref.iloc[:, 1], marker='v', c=c_ref)

    c_test = 'r'
    positive_test, negative_test = divide_to_positive_negative(df_X_test, df_y_test)
    plt.scatter(positive_test.iloc[:, 0], positive_test.iloc[:, 1], marker=',', c=c_test)
    plt.scatter(negative_test.iloc[:, 0], negative_test.iloc[:, 1], marker='v', c=c_test)

    plt.title("Initial data")
    plt.show()
    # pass
