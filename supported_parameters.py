from enum import Enum


Scalers = Enum('Scalers', ['ID', 'MINMAX'])
Encoders = Enum('Encoders', ['EXCLUDE', 'ONEHOT', 'ORDINAL', 'TARGET'])
Distances = Enum('Distances', ['EUCLIDEAN', 'EUCLIDEANSQUARE', 'MANHATTAN', 'CHEBYSHEV', 'CANBERRA'])
