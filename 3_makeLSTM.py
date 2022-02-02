from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from nowprice import access_token
import pandas as pd
import numpy as np
from timechange import *
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import sys


params = {
    "granularity": "M1",
    "count": 1000
}

# デモアカウントでAPI呼び出し
api = API(access_token=access_token)
res = api.request(instruments.InstrumentsCandles(instrument="USD_JPY", params=params))

# DataFrameへ
TIME = [i["time"] for i in res["candles"]]
CLOSE = [i["mid"]["c"] for i in res["candles"]]
data = [TIME, CLOSE]
col = ['time', 'close']
df = pd.DataFrame(columns=col)

for i,v in enumerate(df):
    df[v] = data[i]


print(df.tail())

# 時間表記変更
df['time'] = df['time'].apply(lambda x: to_datetime_japan(x))
# df['time'] = df['time'].apply(lambda x: date_string(x))

df['close'] = df['close'].astype(np.float64)


train = df
del train['time']

test = df

del test['time']

# windowにまとめる
window_len = 10

train_in = []
for i in range(len(train) - window_len):
    temp = train[i:(i + window_len)].copy()
    for col in train:
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

print(test_in)

# PandasからNumpyへ
train_in = [np.array(train_lstm_input) for train_lstm_input in train_in]
train_in = np.array(train_in)

test_in = [np.array(test_lstm_input) for test_lstm_input in test_in]
test_in = np.array(test_in)

print(test_in)
sys.exit()


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
                            epochs=30, batch_size=1, verbose=2, shuffle=True)

# 訓練データから予測をして正解レートと予測したレートをプロット
# fig, ax1 = plt.subplots(1,1)
# ax1.plot(df[df['time']< split_date]['time'][window_len:],
#          train['close'][window_len:], label='Actual', color='blue')
# ax1.plot(df[df['time']< split_date]['time'][window_len:],
#          ((np.transpose(yen_model.predict(train_in))+1) * train['close'].values[:-window_len])[0],
#          label='Predicted', color='red')


# テストデータを使って予測＆プロット
fig, ax1 = plt.subplots(1,1)
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         test['close'][window_len:], label='Actual', color='blue')
ax1.plot(df[df['time']>= split_date]['time'][window_len:],
         ((np.transpose(yen_model.predict(test_in))+1) * test['close'].values[:-window_len])[0],
         label='Predicted', color='red')
ax1.grid(True)

plt.show()

