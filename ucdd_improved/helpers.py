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
