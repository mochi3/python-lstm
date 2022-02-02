import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from nowprice import access_token
import sys
import random
from timechange import *
import itertools
from getData import *


params = {
    "granularity": "M1",
    "count": 300
}

# デモアカウントでAPI呼び出し
api = API(access_token=access_token)
res = api.request(instruments.InstrumentsCandles(instrument="USD_JPY", params=params))

print(res["candles"])


# DataFrameへ
OPEN = [i["mid"]["o"] for i in res["candles"]]
HIGH = [i["mid"]["h"] for i in res["candles"]]
LOW = [i["mid"]["l"] for i in res["candles"]]
CLOSE = [i["mid"]["c"] for i in res["candles"]]
data = [OPEN, HIGH, LOW, CLOSE]
col = ['open', 'high', 'low', 'close']
df = pd.DataFrame(columns=col)

for i,v in enumerate(df):
    df[v] = data[i]

df = df.astype(np.float64)

print(df.tail())

datasize = df.shape[0]

#データの前処理
df.index=range(datasize)

cols=df.columns
print("colsは")
print(cols)

diff_list=[]
#open, closeなどにそれぞれ処理
for col in cols:
    print("colは"+col)
    print("df.loc[:,coll]は", df.loc[:,col])
    print("df.loc[:,coll].pct_change()は", df.loc[:,col].pct_change())
    #diff_dataは変化量？？のため二行目以降を残す（[1:]）
    diff_data=df.loc[:,col].pct_change()[1:]
    print("diff_dataは")
    print(diff_data)
    print("diff_data.index前は", diff_data.index)
    diff_data.index=range(datasize-1)
    print("diff_data.index後は", diff_data.index)
    series = pd.Series(data=diff_data, dtype='float')
    print("seriesは", series)
    diff_list.append(series)

print("seriesを入れたdiff_listは", diff_list)
df=pd.concat(diff_list,axis=1)
print("(df)diff_listをconcatすると", df)


#時間方向を横軸に組み込んだDataFrameの作成
dataframe_list=[df]
print("dataframe_listは", dataframe_list)
wide=3
keys=["{}".format(i) for i in range(wide)]
print("keys=",keys)
for i in range(wide):
    print("dataframe_list[i]=", dataframe_list[i], " (data_kari)dataframe_list[i].drop(i)=",dataframe_list[i].drop(i))
    data_kari=dataframe_list[i].drop(i)
    print("data_kari.index=", data_kari.index)
    print("range(datasize-(i+2))=", range(datasize-(i+2)))
    data_kari.index=range(datasize-(i+2))
    print("data_kari.index=", data_kari.index)
    dataframe_list.append(data_kari)
    print("dataframe_listにdata_kariを追加", data_kari)
    print("dataframe_listのi行目を削除した形？ //ループ終わり")
concat_df=pd.concat(dataframe_list,axis=1,keys=keys).dropna()
print("concat_df=", concat_df)
print(concat_df['0'], concat_df['1'], concat_df['2'])

# makescatter(concat_df.iloc[:,1])

# 学習用データの作成
# 新X
print("preprocessing.scale()は正規化、平均を０にし、標準偏差を１にする")
# .scaleは、まず一行全部抜き出し→標準化→一行を配列に
# そのためo,p,h,lをおしなべて一行に、その後標準化
cd_0 = preprocessing.scale(list(itertools.chain.from_iterable(concat_df['0'].to_numpy())))
print("cd_0", cd_0)
print("cd_0の長さ", len(cd_0))
cd_1 = preprocessing.scale(list(itertools.chain.from_iterable(concat_df['1'].to_numpy())))
cd_2 = preprocessing.scale(list(itertools.chain.from_iterable(concat_df['2'].to_numpy())))
XX = np.stack([cd_0, cd_1, cd_2])
print("XX=",XX)

sys.exit()

#参考元のX、保留
print("preprocessing.scale(concat_df)", preprocessing.scale(concat_df).astype(np.float64))
print(len(preprocessing.scale(concat_df).astype(np.float64)))
print(len(preprocessing.scale(concat_df).astype(np.float64)[0]))
X=preprocessing.scale(concat_df).astype(np.float64)[1:,1]
print("X=",X)
print("len(X)", len(X))

print("もしfが、〜-0.01なら0、　-0.01〜0.01なら1、　0.01〜なら2")
f=lambda x: 2 if x>0.00003   else    0 if x<-0.00003    else 1
print(".iroc[:,1]は、一列目抜き出し")
print("concat_df.iroc[:,1]=", concat_df.iloc[:,1])
print(".mapは、seriesの各要素に関数適用")
print("concat_df.iloc[:,1].map(f)=", concat_df.iloc[:,1].map(f))
print(".valuesは辞書型の値を取得")
print("concat_df.iloc[:,1].map(f).values.astype(np.int64)=", concat_df.iloc[:,1].map(f).values.astype(np.int64))
print("上の長さ", len(concat_df.iloc[:,1].map(f).values.astype(np.int64)))
preY = concat_df.iloc[:,1].map(f).values.astype(np.int64)
Y=preY[:preY.shape[0]-1]
print("Yの長さ", len(Y))

print("XXの長さ", len(XX[0]), len(XX[1]))
sys.exit()

train_X,test_X,train_y,test_y=train_test_split(XX,Y,random_state=0)
print("train_X")
print(train_X)
print("test_x")
print(test_X)
print("train_y")
print(train_y)
print("test_y")
print(test_y)


C_list = [10 ** i for i in range(-5, 7)]
print("C_list", C_list)

# グラフ描画用の空リストを用意
train_accuracy = []
test_accuracy = []
print([train_X])


for C in tqdm(C_list):
    model = SVC(C=C)

    model.fit(train_X, train_y)
    train_accuracy.append(model.score(train_X, train_y))
    test_accuracy.append(model.score(test_X, test_y))

predict_y=model.predict(test_X)


# グラフの準備
plt.semilogx(C_list, train_accuracy, label="accuracy of train_data")
plt.semilogx(C_list, test_accuracy, label="accuracy of test_data")
plt.title("accuracy with changing C")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend()
plt.show()


print("Average score is {}".format(np.mean(test_accuracy)))
print("Max score is {}".format(np.max(test_accuracy)))

def kentei(predict_y,test_y):
  count=0
  for i in range(len(predict_y)):
    if predict_y[i]==2 and test_y[i]==0:
      count+=1
  return count/predict_y.tolist().count(2)

print("投資失敗率:{}".format(kentei(predict_y,test_y)))