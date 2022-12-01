from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

import accepting
import my_preprocessing
import ucdd
import ucdd_eval

if __name__ == '__main__':
    ucdd_eval.evaluate_ucdd()


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
