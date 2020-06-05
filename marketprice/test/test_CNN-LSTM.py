import context
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from test_sequence import sale
from modules.data_preparation import split_sequence

data = sale.to_numpy().flatten()

n_steps_in, n_steps_out = 14, 7

X, y = split_sequence(data, n_steps_in, n_steps_out)
# X.shape is [samples, timesteps]
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 7

X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

# define CNN-LSTM model
model = Sequential()
model.add(TimeDistributed(Conv1D(64, 1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D()))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

X_test = np.array(data[-n_steps_in:]).reshape(1, n_seq, n_steps, n_features)

y_hat = model.predict(X_test, verbose=1)

print(f'y = {y[-n_steps_out:]}, y_hat = {y_hat}')

print(model.summary)