import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import sys
import random
from timechange import *
import itertools
from getData import *
import math
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def svr_1(df):
    # df = get_now_data(300, 'M5')

    df_r = df['close']-df['open']
    df_h = df['high']-df['low']
    df_x = pd.concat([df_r, df_h], axis=1)
    # df_x = df_x.applymap(lambda x: -1 if x < -0.001 else 1 if x > 0.001 else 0)
    # df_x = df_x.applymap(lambda x: -1 if x < 0 else 1)
    df_x = df_x.applymap(lambda x: 0.0001 if x == 0 else x)


    df_x = pd.concat([df_x.pct_change(periods=1)], axis=1)
    # df_x = pd.concat([df_x], axis=1)
    # print(df_x)
    df_x1 = df['open'].pct_change(periods=1)
    df_x2 = df['close'].pct_change(periods=1)
    df_x = pd.concat([df_x, df_x.shift(1)], axis=1)
    print(df_x)

    # 標準化
    std = preprocessing.StandardScaler()
    df_x = pd.DataFrame(std.fit_transform(df_x), columns=df_x.columns)


    df_y = df_r.shift(-1)
    # print(df_y)
    # df_y = df_y.map(lambda x: -1 if x < -0.01 else 1 if x > 0.01 else 0)
    df_y = df_y.map(lambda x: -1 if x < 0 else 1)
    # df_y = df_y.map(lambda x: 1 if (x < -0.01) | (x > 0.01) else 0)

    # print(df_y)

    # yとxのサイズ
    df_x = df_x.drop(df_x.index[:5])
    df_y = df_y.drop(df_y.index[:5])
    df_x = df_x.drop(df_x.index[-1:])
    df_y = df_y.drop(df_y.index[-1:])

    # 分割
    # train_x = df_x[:800]
    # train_y = df_y[:800]
    # test_x = df_x[800:]
    # test_y = df_y[800:]

    # print(test_y)
    # df_x = df_x.to_numpy()
    # df_y = df_y.to_numpy()
    print(df_x, df_y)

    # 訓練データ、テストデータに分割
    X, Xtest, y, ytest = train_test_split(df_x, df_y, test_size=0.2, random_state=114514)
    print(X, Xtest, y, ytest)

    # 6:2:2に分割にするため、訓練データのうちの後ろ1/4を交差検証データとする
    # 交差検証データのジェネレーター
    def gen_cv():
        m_train = np.floor(len(y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(y))
        yield (train_indices, test_indices)

    # # 訓練データを基準に標準化（平均、標準偏差で標準化）
    # scaler = StandardScaler()
    # X_norm = scaler.fit_transform(X)
    # # テストデータも標準化
    # Xtest_norm = scaler.transform(Xtest)
    X_norm = X
    print(X_norm)
    print(y)

    # ハイパーパラメータのチューニング
    params_cnt = 2
    params = {"C": np.logspace(-4, 2, 7, base=params_cnt), "epsilon": np.logspace(0, 3, 4, base=params_cnt)}
    gridsearch = GridSearchCV(SVR(), params, cv=3, scoring="r2", return_train_score=True)
    gridsearch.fit(X_norm, y)
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)

    # RBFカーネル、線形、多項式でフィッティング
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr.fit(X, y)
    pre = svr.predict(Xtest)
    print(pre)
    print(len(pre))

    sys.exit()



    print(test_y)
    print(predict_y)

    # グラフの準備
    # plt.semilogx(C_list, train_accuracy, label="accuracy of train_data")
    # plt.semilogx(C_list, test_accuracy, label="accuracy of test_data")
    # plt.title("accuracy with changing C")
    # plt.xlabel("C")
    # plt.ylabel("accuracy")
    # plt.legend()


    y_count = 0
    n_count = 0
    a_count = 0

    print("Average score is {}".format(np.mean(test_accuracy)))
    print("Max score is {}".format(np.max(test_accuracy)))


    for i in range(len(predict_y)):
        if (predict_y[i] == 1) | (predict_y[i] == -1):
            if predict_y[i] == test_y[i]:
                y_count+=1
            # if predict:
            #     n_count+=1
            else:
                n_count+=1

    print("good count=", y_count)
    print("bad count=", n_count)
    print("else count=", a_count)

    # plt.show()


    return test_accuracy

