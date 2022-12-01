# this class is no longer necessary


# import numpy as np
# import pandas as pd
# import sklearn.model_selection
# from sklearn import preprocessing
#
#
# def divide_numeric_categorical(df_x):
#     df_x_numeric = df_x.select_dtypes(include=[np.number])
#     df_x_categorical = df_x.select_dtypes(exclude=[np.number])
#     return df_x_numeric, df_x_categorical
#
#
# def scale_df_with(df, transformer):
#     """Scale numeric df with a given encoder"""
#     df[df.columns] = transformer.fit_transform(df[df.columns])
#     return df
#
#
# def encode_df_with(df, transformer):
#     """Encode categorical df with a given encoder - add and drop columns because of possible one-hot encoding"""
#     original_cols = df.columns
#
#     col = transformer.fit_transform(df[df.columns])
#     df_col = pd.DataFrame(col)
#
#     # make sure all column names are strings
#     df_col.columns = [str(cname) for cname in df_col.columns]
#
#     df = df.join(df_col)
#     df = df.drop(columns=original_cols)
#
#     return df
#
#
# def prepare_data(df_x, scaling, scaler, use_categorical, encoding=False, encoder=None):
#     """Divide df into categorical and numeric data to do possible encoding and scaling"""
#     df_x_num, df_x_cat = divide_numeric_categorical(df_x)
#     if scaling:
#         df_x_num = scale_df_with(df_x_num, scaler)
#
#     if use_categorical:
#         if encoding:
#             df_x_cat = encode_df_with(df_x_cat, encoder)
#         df_x = df_x_num.join(df_x_cat)
#     else:
#         df_x = df_x_num
#
#     return df_x
#
#
# def prepare_data_and_get_batches(df_x, df_y, test_fraction, num_ref_batches, num_test_batches,
#                                  scaling, scaler, use_categorical=False, encoding=False, encoder=None):
#     """Older data preparation and batch creation function doing dataframe operations"""
#     df_x = prepare_data(df_x, scaling, scaler, use_categorical, encoding, encoder)
#
#     df_X_ref, df_X_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
#         df_x, df_y, test_size=test_fraction, shuffle=False)
#
#     X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = get_batches(
#         df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches
#     )
#     return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches
#
#
# def transform_data_and_get_batches(df_x, df_y, test_fraction, num_ref_batches, num_test_batches, transformer):
#     """Better data preparation and batch creation function relying on the existing sklearn's infrastructure"""
#     # if categorical data is transformed using one-hot encoder, USING SPARSE=FALSE IS NECESSARY
#     df_x = pd.DataFrame(transformer.fit_transform(df_x))
#     print('df_x')
#     print(df_x)
#     df_y = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(df_y))
#     print('df y')
#     print(df_y)
#
#     df_X_ref, df_X_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
#         df_x, df_y, test_size=test_fraction, shuffle=False)
#
#     X_ref_batches, y_ref_batches, X_test_batches, y_test_batches = get_batches(
#         df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches
#     )
#     return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches
#
#
# def get_batches(df_X_ref, df_X_test, df_y_ref, df_y_test, num_ref_batches, num_test_batches):
#     """Divide reference and testing data and labels into lists of batches"""
#     X_ref_batches = np.array_split(df_X_ref, num_ref_batches)
#     y_ref_batches = np.array_split(df_y_ref, num_ref_batches)
#     X_test_batches = np.array_split(df_X_test, num_test_batches)
#     y_test_batches = np.array_split(df_y_test, num_test_batches)
#     return X_ref_batches, y_ref_batches, X_test_batches, y_test_batches
#
