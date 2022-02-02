import schedule
import time
from SVM_7 import *
from tqdm import tqdm
import datetime
import getData
from SVM_7_plus import *
from timechange import *
import itertools
from keras_1 import *
from SVR_1 import *


acclist = []

# for i in tqdm(range(0,24)):
#     print(i)
#     end_day = datetime.datetime(2019, 10, 10, 15, 0)
#     end_day = end_day + datetime.timedelta(hours=i)
#     end = datetime.datetime.strftime(end_day, '%Y-%m-%dT%H:%M:%SZ')
#     df = get_past_data(500, "M5", end)
#     acclist.append(svm_7(df))
#
# print(acclist)

# df = get_now_data(500, "M5")
# acclist.append(svm_7(df))

# 回す
def past_times():
    times = []
    acclist = []
    for i in tqdm(range(0, 10)):
        df = get_csv_data(5000)
        time1 = df['time'][len(df)-1]
        # times.append(to_datetime_japan2(time1).weekday())
        times.append(to_datetime_japan2(time1))
        # acc = svm_7(df)[0]
        acclist.append(svm_7(df))

    print(times, acclist)
    acclist = pd.DataFrame(acclist)
    print(acclist)
    print(acclist.mean())
    # print(sum(acclist)/len(acclist))
    # plt.scatter(times, acclist, alpha=0.1)
    # plt.show()


# df_1 = get_csv_data(5000)
df_2 = get_now_data(5000, "M5")
# past_times()
svm_7(df_2)
# svm_7_plus(df_1)
# rnn_1(df_2)
# svr_1(df_2)

# def job():
#     print(1)
#
#
# schedule.every(1).minutes.do(job)
#
#
# while True:
#     schedule.run_pending()
#     time.sleep(1)