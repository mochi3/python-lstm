from datetime import datetime
import pytz
import time
import random
import pandas as pd
import matplotlib.pyplot as plt


def to_datetime_japan(fromm):
    dt = None
    try:
        dt = datetime.strptime(fromm, '%Y-%m-%dT%H:%M:%S.%f000Z')
        dt = pytz.utc.localize(dt).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            dt = datetime.strptime(fromm, '%Y-%m-%dT%H:%M:%S.%f000Z')
            dt = pytz.utc.localize(dt).astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return dt


def to_datetime_japan2(fromm):
    dt = None
    try:
        dt = datetime.strptime(fromm, '%Y-%m-%dT%H:%M:%S')
        dt = pytz.utc.localize(dt).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            dt = datetime.strptime(fromm, '%Y-%m-%dT%H:%M:%S')
            dt = pytz.utc.localize(dt).astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return dt


def date_string(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')


def string_date(str):
    if str is None:
        return ''
    return datetime.strptime(str, '%Y/%m/%d %H:%M:%S')


if __name__ == '__main__':
    start = time.time()
    for i in range(0,11):
        print("a")
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


# def sortrange(x, cutrange):
#     x.sort_values()
#     xx = pd.qcut(x, round(x/cutrange))
#     print("sortrange")
#     print(xx)



def makescatter(x):
    list = [random.random() for i in range(len(x))]
    y = pd.DataFrame(list)
    plt.scatter(x, y, alpha=0.4)
    plt.xlim(-0.0003, 0.0003)
    plt.ylim(-0.5,1.5)
    plt.show()
