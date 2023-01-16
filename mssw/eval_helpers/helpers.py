import numpy as np
import math


def split_to_fixed_size_batches(X, y, batch_size):
    """Split X and y to batches of the given batch_size"""
    chunk_size = batch_size
    print('chunk size', chunk_size)

    num_chunks = math.ceil(X.shape[0] / chunk_size)
    print('number of chunks', num_chunks)
    print('number of data', X.shape[0])
    X_batches = np.array_split(X, num_chunks)
    y_batches = np.array_split(y, num_chunks)

    print('number of resulting batches', len(X_batches))
    print(X_batches[0])
    print(X_batches[0].shape)

    return X_batches, y_batches


def synthetic_data_information(path_to_file):
    data_filename = path_to_file.split('/')[-1]
    type_of_data = path_to_file.split('/')[2].split('_')[0]  # synthetic or real-world
    dataset_name = data_filename.split('_')[0]  # sea, agraw1, agraw2
    drift_type = path_to_file.split('/')[3].split('_')[0]
    drift_width = '0' if drift_type == 'abrupt' else data_filename.split('_')[-1].split('.')[0]
    drift_width = 0.5 if drift_width == '05' else float(drift_width)

    synthetic_filename_info = {
        'data_filename': data_filename,
        'type_of_data': type_of_data,
        'dataset_name': dataset_name,
        'drift_type': drift_type,
        'drift_width': drift_width
    }

    return synthetic_filename_info
