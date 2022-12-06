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
        raise NameError('The scaling', scaler_id, 'is not supported')
    return df_x_ref, df_x_test


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
            raise NameError('The encoding', encoder_id, 'is not supported')
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

    return df_x_ref_final, df_x_test_final


def fpr_and_latency_when_averaging(drift_locations, num_test_batches, true_drift_idx):
    fpr = 0
    latency = 1
    drift_locations_arr = np.array(drift_locations)
    signal_locations_not_before_drift = drift_locations_arr[drift_locations_arr >= true_drift_idx]
    num_batches_with_drift = num_test_batches - true_drift_idx

    if len(drift_locations) >= 1:
        first_drift_location_idx = drift_locations[0]
        if first_drift_location_idx < true_drift_idx:
            fpr = (true_drift_idx - first_drift_location_idx) / true_drift_idx
            if len(signal_locations_not_before_drift) > 0:
                first_location_not_before_drift = signal_locations_not_before_drift[0]
                latency = (first_location_not_before_drift - true_drift_idx) / num_batches_with_drift
        else:
            latency = (first_drift_location_idx - true_drift_idx) / num_batches_with_drift

    return fpr, latency


def detection_fpr(drift_locations, true_drift_idx):
    fpr = []
    if drift_locations != []:
        first_detection_idx = drift_locations[0]
        if first_detection_idx <= true_drift_idx:
            fpr = (true_drift_idx - first_detection_idx) / (true_drift_idx + 1)
    return fpr


def detection_latency(drift_locations, true_drift_idx):
    latency = []
    if drift_locations != []:
        first_detection_idx = drift_locations[0]
        if first_detection_idx >= true_drift_idx:
            num_batches_with_drift = true_drift_idx + 1
            latency = (first_detection_idx - true_drift_idx) / num_batches_with_drift
    return latency


def obtain_preprocessed_batches(
        file_path,
        scaling,
        encoding,
        test_size,
        num_ref_batches,
        num_test_batches
):
    df_x, df_y = accepting.get_clean_df(file_path)

    # convert labels to 0 and 1
    df_y = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(df_y))

    # split data to training and testing (with a joint dataframe)
    df_x_ref, df_x_test, df_y_ref, df_y_test = sklearn.model_selection.train_test_split(
        df_x, df_y, test_size=test_size, shuffle=False)

    # do all the necessary data transformations (e.g. one-hot encoding, scaling)
    # --> might be different for each dataset
    df_x_ref, df_x_test = preprocess_df_x(df_x_ref, df_x_test, df_y_ref, scaling=scaling, encoding=encoding)
    # reindex data to make sure the indices match
    df_y_ref.set_index(df_x_ref.index, inplace=True)
    df_y_test.set_index(df_x_test.index, inplace=True)

    # divide the data in batches
    x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = get_batches(
        df_x_ref, df_x_test, df_y_ref, df_y_test, num_ref_batches=num_ref_batches, num_test_batches=num_test_batches
    )

    return x_ref_batches, y_ref_batches, x_test_batches, y_test_batches


def evaluate_ucdd(
        file_path,
        scaling,
        encoding,
        test_size,
        num_ref_batches,
        num_test_batches,
        random_state,
        additional_check,
        detect_all_training_batches,
        metric_id,
        only_first_drift=False,
        use_pyclustering=True,
        true_drift_idx=2,
        debug=False
):
    # do all preprocessing and obtain the final batches
    x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = obtain_preprocessed_batches(
        file_path,
        scaling,
        encoding,
        test_size,
        num_ref_batches,
        num_test_batches
    )

    # use ucdd on the batched data and find drift locations
    drift_locations = ucdd_pyclustering.drift_occurrences_list(
        x_ref_batches,
        x_test_batches,
        random_state=random_state,
        additional_check=additional_check,
        detect_all_training_batches=detect_all_training_batches,
        only_first_drift=only_first_drift,
        metric_id=metric_id,
        debug=debug
    )
    print('drift locations', drift_locations)

    return drift_locations


def evaluate_ucdd_until_convergence(
        file_path,
        scaling,
        encoding,
        test_size,
        num_ref_batches,
        num_test_batches,
        additional_check,
        detect_all_training_batches,
        metric_id,
        only_first_drift=False,
        use_pyclustering=True,
        min_runs=5,
        max_runs=50,
        std_threshold=0.1,
        highest_fq_threshold=0.5,
        true_drift_idx=2,
        debug=False
):
    # do all preprocessing and obtain the final batches
    x_ref_batches, y_ref_batches, x_test_batches, y_test_batches = obtain_preprocessed_batches(
        file_path,
        scaling,
        encoding,
        test_size,
        num_ref_batches,
        num_test_batches
    )

    random_state = 0
    drift_locations_multiple_runs = []
    detection_std = 1.0
    highest_frequency = 0.0
    while random_state < max_runs and detection_std > std_threshold and highest_frequency < highest_fq_threshold:
        drift_locations = ucdd_pyclustering.drift_occurrences_list(
            x_ref_batches,
            x_test_batches,
            random_state=random_state,
            additional_check=additional_check,
            detect_all_training_batches=detect_all_training_batches,
            only_first_drift=only_first_drift,
            metric_id=metric_id,
            debug=debug
        )
        drift_locations_multiple_runs.append(drift_locations)
        random_state += 1

        nonempty_drift_locations = []
        if random_state >= min_runs:
            nonempty_drift_locations = [lst for lst in drift_locations_multiple_runs if len(lst) > 0]
            if len(nonempty_drift_locations) > 0:
                normalised_drift_locations = np.array(nonempty_drift_locations) / (num_test_batches - 1)
                # when drift was detected at least once, get the standard deviation of drift signal locations
                detection_std = np.std(normalised_drift_locations)
                nonempty_drift_locations_1d = np.ndarray.flatten(np.array(nonempty_drift_locations))
                highest_frequency = np.amax(np.bincount(nonempty_drift_locations_1d)) / len(nonempty_drift_locations_1d)
            else:
                # if drift was not detected in either of the initial runs, assume it won't ever be detected
                detection_std = 0.0

        print('drift_locations_multiple_runs', drift_locations_multiple_runs)
        print('nonempty_drift_locations', nonempty_drift_locations)
        print('detection_std', detection_std)
        print('highest_frequency', highest_frequency)
    # nonempty_drift_locations = [lst for lst in drift_locations_multiple_runs if len(lst) > 0]
    # if len(nonempty_drift_locations) > 0:
    #     # when drift was detected at least once, get the standard deviation of drift signal locations
    #     detection_std = np.std(nonempty_drift_locations)
    # else:
    #     # if drift was never detected in all the initial runs, assume it won't ever be detected
    #     detection_std = 0.0
    #
    # print('detection std after min_runs', detection_std)
    #
    # while random_state < max_runs and detection_std > std_threshold:
    #     drift_locations = ucdd_pyclustering.drift_occurrences_list(
    #         x_ref_batches,
    #         x_test_batches,
    #         random_state=random_state,
    #         additional_check=additional_check,
    #         detect_all_training_batches=detect_all_training_batches,
    #         only_first_drift=only_first_drift,
    #         metric_id=metric_id,
    #         debug=debug
    #     )
    #     drift_locations_multiple_runs.append(drift_locations)
    #     random_state += 1
    #     nonempty_drift_locations = [lst for lst in drift_locations_multiple_runs if len(lst) > 0]
    #     if len(nonempty_drift_locations) > 0:
    #         detection_std = np.std(nonempty_drift_locations)
    #     print('current detection_std', detection_std)
    return drift_locations_multiple_runs


def evaluate_ucdd_multiple_random_states(file_path, scaling, encoding, test_size, num_ref_batches, num_test_batches,
                                         random_states, additional_check, detect_all_training_batches,
                                         only_first_drift,
                                         debug=False, use_pyclustering=False, metric_id=spms.Distances.EUCLIDEAN):
    drift_locations_multiple_runs = []
    for random_state in random_states:
        drift_locations = evaluate_ucdd(
            file_path,
            scaling,
            encoding,
            test_size,
            num_ref_batches,
            num_test_batches,
            random_state,
            additional_check,
            detect_all_training_batches,
            metric_id=metric_id,
            only_first_drift=only_first_drift,
            use_pyclustering=use_pyclustering,
            debug=debug
        )
        drift_locations_binary = np.repeat(False, num_test_batches)
        drift_locations_binary[drift_locations] = True
        drift_locations_multiple_runs.append(list(drift_locations_binary))

    return drift_locations_multiple_runs
