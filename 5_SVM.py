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


# df = get_past_data(2000)
df = get_now_data_volume(300)
# 欠損値
# print(df.isnull().sum())

# 差分（１０分前）を出す
# df_s = df.diff(periods=10)
# df_s = df_s.drop(df_s.index[range(0,10)])
# print(df_s.head(n=30))
# plt.hist(x=df_s['close'], range=(-0.1, 0.1), bins=200)
# plt.show()
# print(100*( len(df[(df_s['close'] > 0.03) | (df_s['close'] < -0.0)]) /len(df)))

# 変化率（1,2,3分前）
df_nv = df.drop(columns=['volume'])
df_pc1 = df_nv.pct_change(periods=1)
df_pc2 = df_nv.pct_change(periods=2)
df_pc3 = df_nv.pct_change(periods=3)
df_pc1 = df_pc1.to_numpy()
print(df_pc1)
sys.exit()
df_pc = pd.concat([df_pc1, df_pc2, df_pc3, df['volume']], axis=1)
# df_pc = df_pc.drop(index=df_pc.index[:10])
# # ユーロをつける
# df2 = df2.drop(columns='volume')
# df_pc = pd.concat([df_pc, df2.pct_change(periods=1)], axis=1)
# print(df_pc)

# 標準化
std = preprocessing.StandardScaler()
df_pc = pd.DataFrame(std.fit_transform(df_pc), columns=df_pc.columns)

# yを作る
# 10分後の差分
df_s = df['close'].diff(periods=-10)
df_y = df_s.map(lambda x: -1 if x < -0.02 else 1 if x > 0.02 else 0)
# df_y = df_y.drop(df_y.index[-10:])

# xとyの長さを合わせる
df_x = df_pc
df_x = df_pc.drop(df_x.index[-10:])
df_y = df_y.drop(df_y.index[-10:])
df_x = df_x.drop(df_x.index[:3])
df_y = df_y.drop(df_y.index[:3])

# # 特徴量の重要度を出す
# feat_labels = df_x.columns
# forest = RandomForestClassifier(n_estimators=500, random_state=1)
# forest.fit(df_x, df_y)
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]
# plt.bar(range(df_x.shape[1]), importances[indices], align='center')
# plt.xticks(range(df_x.shape[1]), feat_labels[indices], rotation=90)
# plt.xlim([-1, df_x.shape[1]])
# plt.tight_layout()
# plt.show()


# # volumeはいらなそうなので外す
# df_x = df_pc.drop(columns='volume')

# 分割
train_x = df_x[:250]
train_y = df_y[:250]
test_x = df_x[250:]
test_y = df_y[250:]

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
p_count = 0
a_count = 0
print(predict_y)
print(test_y)

for i in range(len(predict_y)):
    if (predict_y[i]==1 & test_y[i]==1) | (predict_y[i]==-1 & test_y[i]==-1):
        y_count+=1
    if (predict_y[i]==-1 & test_y[i]==1) | (predict_y[i]==1 & test_y[i]==-1):
        n_count+=1
    if (predict_y[i] == 1 & test_y[i] == 0) | (predict_y[i] == -1 & test_y[i] == 0):
        p_count += 1
    else:
        a_count+=1

print("1=1, -1=-1 count=", y_count)
print("1=-1, -1=1 count=", n_count)
print("1or-1 = 0 count=", p_count)
print("0 = 1or-1 count=", a_count)


print("Average score is {}".format(np.mean(test_accuracy)))
print("Max score is {}".format(np.max(test_accuracy)))

plt.show()


# def kentei(predict_y, test_y):
#     count=0
#     for i in range(len(predict_y)):
#         if predict_y[i]==2 and test_y[i]==0:
#           count+=1
#     return count/predict_y.tolist().count(2)
#
# print("投資失敗率:{}".format(kentei(predict_y, test_y)))

