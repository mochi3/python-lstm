import dill
import numpy as np
import datetime
import matplotlib.pyplot as plt
from timechange import *

dill.load_session('makeLSTM.pkl')

print(df[df['time']< split_date])
print(df[df['time']< split_date]['time'][window_len:])
mm = df[df['time']< split_date]['time'][window_len:]

for t in mm:
    t = string_date(t)
    print(t)

print(mm)

# 訓練データから予測をして正解レートと予測したレートをプロット
fig, ax1 = plt.subplots(1,1)
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