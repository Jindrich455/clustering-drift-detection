# This is a sample Python script.
import preprocessing
import ucdd


dataset_paths = {
    'drift_2d': 'datasets/drift_2d.arff',
    'sea_abrupt': 'datasets/sea_1_abrupt_drift_0_noise_balanced.arff',
    'agraw1_abrupt': 'datasets/agraw1_1_abrupt_drift_0_noise_balanced.arff',
    'agraw2_abrupt': 'datasets/agraw2_1_abrupt_drift_0_noise_balanced.arff'
}


if __name__ == '__main__':
    df_reference, df_test = preprocessing.get_pandas_reference_testing(dataset_paths['drift_2d'], 0.5)
    ref_batch_list, test_batch_list = preprocessing.divide_to_batches(df_reference, 1, df_test, 1)
    preprocessing.print_batch_info(ref_batch_list, 'reference batches')
    preprocessing.print_batch_info(test_batch_list, 'testing batches')
    drift_happening = ucdd.detect_cd(ref_batch_list[0], test_batch_list[0])
    if drift_happening:
        print('DRIFT!!!!')
