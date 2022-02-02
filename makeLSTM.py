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


# データの取得
df = pd.read_csv("USDJPY_20191018_1Y.csv" ,header=None,usecols=[0,2,3,4,5,6])
df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
# df.index = 'time'
# df = df.set_index('time')
# print(df.head())

# 時間表記変更
df['time'] = df['time'].apply(lambda x: to_datetime_japan2(x))
df['time'] = df['time'].apply(lambda x: date_string(x))


# データの分割
split_date = '2018/10/19 19:13:00'
train, test = df[df['time'] < split_date], df[df['time'] >= split_date]
del train['time']
del test['time']

print(train)

# windowにまとめる
window_len = 10

train_in = []
for i in range(len(train) - window_len):
    temp = train[i:(i + window_len)].copy()
    for col in train:
        print(type(temp[col].iloc[0]))
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    train_in.append(temp)
train_out = (train['close'][window_len:].values / train['close'][:-window_len].values) - 1

test_in = []
for i in range(len(test) - window_len):
    temp = test[i:(i + window_len)].copy()
    for col in test:
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    test_in.append(temp)
test_out = (test['close'][window_len:].values / test['close'][:-window_len].values) - 1

sys.exit()

# PandasからNumpyへ
train_in = [np.array(train_lstm_input) for train_lstm_input in train_in]
train_in = np.array(train_in)

test_in = [np.array(test_lstm_input) for test_lstm_input in test_in]
test_in = np.array(test_in)


# LSTMのモデルを設定
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


# ランダムシードの設定
np.random.seed(202)

# 初期モデルの構築
yen_model = build_model(train_in, output_size=1, neurons=20)

# データを流してフィッティングさせましょう
yen_history = yen_model.fit(train_in, train_out,
                            epochs=50, batch_size=1, verbose=2, shuffle=True)

# dill.dump_session('makeLSTM.pkl')

