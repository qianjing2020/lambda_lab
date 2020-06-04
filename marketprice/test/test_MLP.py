import numpy as np
import pandas as pd

import context
from modules.data_preparation import split_sequence

from test_sequence import sale

#data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data = sale.to_numpy().flatten()
#print(data)


X, y = split_sequence(data, n_steps=3)

## transform input from [samples, features] to [samples, timesteps, features], a step for CNN or LSTM 
X = X.reshape((X.shape[0], X.shape[1], 1))

print(X.shape, y.shape) # show each sample
for i in range(len(X)):
    print(X[i], y[i])

print('Good!')