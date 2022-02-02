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


df = get_past_data(3000)
# df = get_now_data_volume(800)
# 欠損値
# print(df.isnull().sum())

# ローソク
df_r = df['close']-df['open']
df_v = df_r.copy()
# print(df_r.head(n=30))
print(df)

# ひげの長さ
df_h = (abs(df['high']-df['low'])-abs(df['close']-df['open']))
print(df_h)

for i in range(0, len(df_r)-5):
    df_v.loc[i] = 0
    if (df_r.loc[i] < -0.002) & (df_r.loc[i+1] < -0.002) & (df_r.loc[i+1] < -0.002):
        df_v.loc[i] = 1
    if (df_r.loc[i] > 0.002) & (df_r.loc[i+1] > 0.002) & (df_r.loc[i+1] > 0.002):
        df_v.loc[i] = 1
df_y = df_v
print(df_v.tail(n=30))

df_nv = df.drop(columns=['volume', 'high', 'low'])
df_pc1 = df_nv.pct_change(periods=1)
df_pc2 = df_nv.pct_change(periods=2)

df_pc = pd.concat([df_pc1, df_pc2, df_r, df_h], axis=1)

# df_pc = df_pc.drop(index=df_pc.index[:10])
print(df_pc)

# 標準化
std = preprocessing.StandardScaler()
df_pc = pd.DataFrame(std.fit_transform(df_pc), columns=df_pc.columns)

# yとxのサイズ
df_x = df_pc
df_x = df_pc.drop(df_x.index[:3])
df_y = df_y.drop(df_y.index[:3])
df_x = df_x.drop(df_x.index[-5:])
df_y = df_y.drop(df_y.index[-5:])

print(df_x.isnull().sum(), df_y.isnull().sum())

# 分割
train_x = df_x[:2500]
train_y = df_y[:2500]
test_x = df_x[2500:]
test_y = df_y[2500:]

# modelへ
C_list = [10 ** i for i in range(-5, 5)]
print("C_list", C_list)
train_accuracy = []
test_accuracy = []

for C in tqdm(C_list):
    model = SVC(C=C)

    model.fit(train_x, train_y)
    train_accuracy.append(model.score(train_x, train_y))
    test_accuracy.append(model.score(test_x, test_y))

predict_y=model.predict(test_x)

# print(predict_y)
# sys.exit()

# グラフの準備
plt.semilogx(C_list, train_accuracy, label="accuracy of train_data")
plt.semilogx(C_list, test_accuracy, label="accuracy of test_data")
plt.title("accuracy with changing C")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend()

test_y = test_y.to_numpy()

y_count = 0
n_count = 0
a_count = 0
print(predict_y)
print(test_y)

# test_y = test_y.to_numpy()
#
for i in range(len(predict_y)):
    if (predict_y[i] == 1) & (test_y[i] == 1):
        y_count+=1
    if (predict_y[i] == 1) & (test_y[i] == 0):
        n_count+=1
    else:
        a_count+=1

print("good count=", y_count)
print("bad count=", n_count)
print("1or-1 = 0 or 0 = 1or-1 count=", a_count)


print("Average score is {}".format(np.mean(test_accuracy)))
print("Max score is {}".format(np.max(test_accuracy)))

plt.show()
