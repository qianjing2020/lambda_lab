import context
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from test_sequence import sale
from modules.data_preparation import split_sequence

data = sale.to_numpy().flatten()

n_steps_in, n_steps_out = 14, 7

X, y = split_sequence(data, n_steps_in, n_steps_out)

# transform input from [samples, features] to [samples, timesteps, features], a step needed for CNN or LSTM 
n_features = 1
X= X.reshape((X.shape[0], X.shape[1], n_features))

print(X.shape, y.shape) # show each sample
for i in range(len(X)):
    print(X[i], y[i])

# CNN model
model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D())
model.add(Conv1D(32, 2, activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=1)

X_test = np.array(data[-n_steps_in:]).reshape(1, n_steps_in, n_features)

y_hat = model.predict(X_test, verbose=0)

print(f'y = {y[-n_steps_out:]}, y_hat = {y_hat}')

print(model.summary)