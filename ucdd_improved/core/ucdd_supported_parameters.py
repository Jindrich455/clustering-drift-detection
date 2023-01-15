from enum import Enum


Encoders = Enum('Encoders', ['EXCLUDE', 'ONEHOT', 'ORDINAL', 'TARGET'])
TrainBatchStrategies = Enum('TrainBatchStrategies', ['ANY', 'SUBMAJORITY', 'MAJORITY', 'ALL'])
