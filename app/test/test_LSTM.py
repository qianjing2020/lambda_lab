import context
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from test_sequence import sale
from modules.data_preparation import split_sequence

data = sale.to_numpy().flatten()

n_steps_in, n_steps_out = 14, 7

X, y = split_sequence(data, n_steps_in, n_steps_out)

# transform input from [samples, features] to [samples, timesteps, features], a step needed for CNN or LSTM 
n_features = 1
X= X.reshape((X.shape[0], X.shape[1], n_features))


# define model
model = Sequential()
model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
#stack one more LSTM
model.add(LSTM(40, activation='relu'))
model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
# fit model
model.fit(X, y, epochs=500, verbose=1)

X_test = np.array(data[-n_steps_in:]).reshape(1, n_steps_in, n_features)

y_hat = model.predict(X_test, verbose=1)

print(f'y = {y[-n_steps_out:]}, y_hat = {y_hat}')

print(model.summary)