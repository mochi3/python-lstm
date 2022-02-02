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


def svm_7_plus(df):
    # df = get_now_data(300, 'M5')

    df_r = df['close']-df['open']
    df_h = df['high']-df['low']
    df_x = pd.concat([df_r, df_h], axis=1)
    # df_x = df_x.applymap(lambda x: -1 if x < -0.001 else 1 if x > 0.001 else 0)
    # df_x = df_x.applymap(lambda x: -1 if x < 0 else 1)
    df_x = df_x.applymap(lambda x: 0.0001 if x == 0 else x)

    df_x = pd.concat([df_x.diff(periods=1), df_x.diff(periods=2), df_x.diff(periods=3), df_x.diff(periods=4)], axis=1)
    df_x = pd.concat([df_x], axis=1)

    # 標準化
    std = preprocessing.StandardScaler()
    df_x = pd.DataFrame(std.fit_transform(df_x), columns=df_x.columns)


    df_y = df_r.shift(-1)
    # df_y = df_y.map(lambda x: -1 if x < -0.01 else 1 if x > 0.01 else 0)
    # df_y = df_y.map(lambda x: -1 if x < 0 else 1)
    df_y = df_y.map(lambda x: 1 if (x < -0.01) | (x > 0.01) else 0)


    # yとxのサイズ
    df_x = df_x.drop(df_x.index[:5])
    df_y = df_y.drop(df_y.index[:5])
    df_x = df_x.drop(df_x.index[-1:])
    df_y = df_y.drop(df_y.index[-1:])

    # 分割
    train_x = df_x[:4500]
    train_y = df_y[:4500]
    test_x = df_x[4500:]
    test_y = df_y[4500:]


    # modelへ
    C_list = [10 ** i for i in range(0, 1)]
    # g_list = [10 ** i for i in range(-5,10)]

    train_accuracy = []
    test_accuracy = []

    for C in C_list:
        model = SVC(C=C)

        model.fit(train_x, train_y)
        train_accuracy.append(model.score(train_x, train_y))
        test_accuracy.append(model.score(test_x, test_y))

    # for C in tqdm(C_list):
    #     model = SVC(C=C)
    #
    #     model.fit(train_x, train_y)
    #     train_accuracy.append(model.score(train_x, train_y))
    #     test_accuracy.append(model.score(test_x, test_y))

    # for g in tqdm(g_list):
    #     model = SVC(C=10 ** 3 , gamma=g)
    #
    #     model.fit(train_x, train_y)
    #     train_accuracy.append(model.score(train_x, train_y))
    #     test_accuracy.append(model.score(test_x, test_y))


    predict_y=model.predict(test_x)
    test_y = test_y.to_numpy()

    print("score is {}".format(np.max(test_accuracy)))

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

    print(len(predict_y))
    for i in range(len(predict_y)):
        if (predict_y[i] == 1) & (predict_y[i] == test_y[i]):
            y_count+=1
        # if predict:
        #     n_count+=1
        if (predict_y[i] == 1) & (predict_y[i] != test_y[i]):
            n_count+=1

    print("good count=", y_count)
    print("bad count=", n_count)
    count_score = y_count/(y_count+n_count)
    print("count score=", count_score)

    # plt.show()

    return count_score

