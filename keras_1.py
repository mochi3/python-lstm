from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from getData import *
from sklearn import preprocessing
import sys

def rnn_1(df):
    '''
    データ準備
    '''
    np.random.seed(0)  # 乱数を固定値で初期化し再現性を持たせる

    # df = get_now_data(500, "M5")
    df_r = df['close'] - df['open']
    df_h = df['high'] - df['low']
    df_x = pd.concat([df_r, df_h], axis=1)
    # df_x = df_x.applymap(lambda x: -1 if x < -0.001 else 1 if x > 0.001 else 0)
    # df_x = df_x.applymap(lambda x: -1 if x < 0 else 1)
    df_x = df_x.applymap(lambda x: 0.0001 if x == 0 else x)

    df_x = pd.concat([df_x.pct_change(periods=1), df_x.pct_change(periods=2), df_x.pct_change(periods=3)], axis=1)
    # df_x = df_x.diff(periods=1)
    # df_x = pd.concat([df_x], axis=1)
    df_x = df['open'].pct_change(periods=1)
    df_x = pd.concat([df_x, df_x.shift(1)], axis=1)


    # 標準化
    std = preprocessing.StandardScaler()
    df_x = pd.DataFrame(std.fit_transform(df_x), columns=df_x.columns)

    df_y = df_r.shift(-1)
    # df_y = df_y.map(lambda x: 0 if x < -0.01 else 2 if x > 0.01 else 1)
    df_y = df_y.map(lambda x: 0 if x < 0 else 1)


    # yとxのサイズ
    df_x = df_x.drop(df_x.index[:10])
    df_y = df_y.drop(df_y.index[:10])
    df_x = df_x.drop(df_x.index[-1:])
    df_y = df_y.drop(df_y.index[-1:])

    # # 分割
    # train_x = df_x[:4500]
    # train_y = df_y[:4500]
    # test_x = df_x[4500:]
    # test_y = df_y[4500:]
    X = df_x
    T = df_y

    T = np_utils.to_categorical(T)
    # 数値を、位置に変換 [0,1,2] ==> [ [1,0,0],[0,1,0],[0,0,1] ]
    train_x, test_x, train_t, test_t = train_test_split(X, T, train_size=0.9, test_size=0.1)  # 訓練とテストで分割

    '''
    モデル作成
    '''
    model = Sequential()
    model.add(Dense(input_dim=2, units=2)) # 4次元を3種に分類
    model.add(Activation('sigmoid'))  # softmax
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

    '''
    トレーニング
    '''
    model.fit(train_x, train_t, epochs=100, batch_size=32)

    '''
    学習済みモデルでテストデータで分類する
    '''
    Y = model.predict_classes(test_x, batch_size=32)

    '''
    結果検証
    '''
    _, T_index = np.where(test_t > 0)  # to_categorical の逆変換
    print()
    print('RESULT')
    print(Y)
    print(T_index)
    print(Y == T_index)
    print("true=", sum(Y == T_index))
    print(sum(Y == T_index)/len(Y))

    return sum(Y == T_index)/len(Y)