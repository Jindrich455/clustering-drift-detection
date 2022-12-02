from enum import Enum


Scalers = Enum('Scalers', ['ID', 'MINMAX'])
Encoders = Enum('Encoders', ['ID', 'EXCLUDE', 'ONEHOT', 'ORDINAL', 'TARGET'])
Distances = Enum('Distances', ['EUCLIDEAN', 'EUCLIDEAN_SQUARE', 'MANHATTAN', 'CHEBYSHEV', 'CANBERRA'])
