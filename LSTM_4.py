import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

import matplotlib.pyplot as plt
import pickle
import dill
from timechange import *
import sys

def lstm_4(df):

    def build_model(inputs, outputs, neurons, activ_func,
                    dropout=0.1, loss="mean_squared_error"):
        model = Sequential()

        model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dropout(dropout))
        # model.add(Dense(units=outputs.shape[1]))
        model.add(Dense(units=outputs.shape[1],
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Activation(activ_func))
        model.compile(loss="categorical_crossentropy",
                      optimizer="RMSprop", metrics=['categorical_accuracy'])

        return model


    model = build_model(train_x_data, train_y_data, neurons=20, activ_func="softmax")

    history = model.fit(train_x_data, train_y_data,
                        epochs=100, batch_size=32, verbose=1, validation_split=0.2)