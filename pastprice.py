import json
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20.exceptions import V20Error
from nowprice import access_token
import matplotlib.pyplot as plt
import numpy as np
from timechange import *
from datetime import datetime
import pytz
from csvcontrol import *
import numba


def oanda():
    api = API(access_token=access_token, environment="practice")
    return api


def listoanda():

    api = oanda()

    start_day = datetime(2019, 10, 17)
    end_day = datetime(2019, 10, 18)
    start = datetime.strftime(start_day, '%Y-%m-%dT%H:%M:%SZ')
    end = datetime.strftime(end_day, '%Y-%m-%dT%H:%M:%SZ')

    params = {
        #"count": 3,  # 足3本取得
        "granularity": "M1",
        "price": "B",  # Bidを取得
        "from": start,
        "to": end,
        #"time": "2019-09-03",
    }

    res = []
    for r in InstrumentsCandlesFactory(instrument="USD_JPY", params=params):
        api.request(r)
        res.extend(r.response.get('candles'))

    return res


def createndarray():
    res = listoanda()
    arr1 = np.arange(12).reshape(2, 6)
    for c in res:
        arr1 = np.append(arr1, [[c['time'], c['volume'], float(c['bid']['o']), float(c['bid']['h']),
                                float(c['bid']['l']), float(c['bid']['c'])]], axis=0)
    arr2 = np.delete(arr1, [0,1], 0)
    return arr2


arr2 = createndarray()

np.savetxt('usd_jpy_20191018ss.csv', arr2, delimiter=',', fmt='%s')

# try:
#     api.request(instruments_candles)
#     response = instruments_candles.response
#     #print(json.dumps(response, indent=4))
#     print(response)
#
# except V20Error as e:
#     print("Error: {}".format(e))


# plt.plot(time1,bid1)
# plt.ylim(min(bid1),max(bid1))
# #plt.show()
