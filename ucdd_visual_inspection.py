from matplotlib import pyplot as plt
from pandas import Series
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import accepting
from sklearn.compose import make_column_selector as selector

import my_preprocessing
import ucdd


def show_ucdd():
    df_x, df_y = accepting.get_clean_df('tests/test_datasets/drift_2d.arff')

    transformer = ColumnTransformer([
        ('num', MinMaxScaler(), selector(dtype_include='number')),
    ])

    X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = my_preprocessing.transform_data_and_get_batches(
        df_x, df_y, test_fraction=0.5, num_ref_batches=1, num_test_batches=1, transformer=transformer
    )

    show_initial_data(X_ref_batches, y_ref_batches, X_test_batches, y_test_batches)

    drift_occurrences = ucdd.drift_occurrences_list(X_ref_batches, X_test_batches, random_state=0, show_2d_plots=True)



def divide_to_positive_negative(df_X, df_y):
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
    plt.scatter(positive_ref.iloc[:, 0], positive_ref.iloc[:, 1], marker=',', c=c_ref)
    plt.scatter(negative_ref.iloc[:, 0], negative_ref.iloc[:, 1], marker='v', c=c_ref)

    c_test = 'r'
    positive_test, negative_test = divide_to_positive_negative(df_X_test, df_y_test)
    plt.scatter(positive_test.iloc[:, 0], positive_test.iloc[:, 1], marker=',', c=c_test)
    plt.scatter(negative_test.iloc[:, 0], negative_test.iloc[:, 1], marker='v', c=c_test)

    plt.title("Initial data")
    plt.show()
    # pass


show_ucdd()
