import context
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from test_sequence import sale
from modules.data_preparation import split_sequence

data = sale.to_numpy().flatten()

n_steps_in, n_steps_out = 3, 2

X, y = split_sequence(data, n_steps_in, n_steps_out)

## transform input from [samples, features] to [samples, timesteps, features], a step for CNN or LSTM 
#X = X.reshape((X.shape[0], X.shape[1], 1))

print(X.shape, y.shape) # show each sample
for i in range(len(X)):
    print(X[i], y[i])

# MLP model
MLP = Sequential()
MLP.add(Dense(100, activation='relu', input_dim=n_steps_in))
MLP.add(Dense(10, activation='relu'))
MLP.add(Dense(n_steps_out))
MLP.compile(optimizer='adam', loss='mse')

MLP.fit(X, y, epochs=2000, verbose=0)

X_test = np.array(data[-n_steps_in:]).reshape(1, n_steps_in)
y_hat = MLP.predict(X_test, verbose=1)

print(f'y = {y[-n_steps_out:]}, y_hat = {y_hat}')

