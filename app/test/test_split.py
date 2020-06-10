import context
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from test_sequence import sale
from modules.data_preparation import split_sequence

data = sale.to_numpy().flatten()

# split data into features and target
n_steps_in, n_steps_out = 3, 1
X, y = split_sequence(data, n_steps_in, n_steps_out)

tscv = TimeSeriesSplit(n_splits=5, max_train_size=None)
print(tscv)

for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]