# MLP model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

MLP = Sequential()
MLP.add(Dense(100, activation='relu', input_dim=n_steps))
MLP.add(Dense(1))
MLP.compile(optimizer='adam', loss='mse')

