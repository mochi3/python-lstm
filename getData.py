from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from nowprice import access_token
import pandas as pd
import numpy as np
import datetime
import random


def get_now_data(count, gran):
    params = {
        "granularity": gran,
        "count": count
    }

    # デモアカウントでAPI呼び出し
    api = API(access_token=access_token)
    res = api.request(instruments.InstrumentsCandles(instrument="USD_JPY", params=params))
    print(res['candles'][0]['time'], res['candles'][count-1]['time'])
    print(res['candles'][count-1]["mid"]["o"], res['candles'][count-1]["mid"]["c"])

    # DataFrameへ
    TIME = [i["time"] for i in res["candles"]]
    OPEN = [i["mid"]["o"] for i in res["candles"]]
    HIGH = [i["mid"]["h"] for i in res["candles"]]
    LOW = [i["mid"]["l"] for i in res["candles"]]
    CLOSE = [i["mid"]["c"] for i in res["candles"]]
    data = [TIME, OPEN, HIGH, LOW, CLOSE]
    col = ['time', 'open', 'high', 'low', 'close']
    df = pd.DataFrame(columns=col)


    for i,v in enumerate(df):
        df[v] = data[i]

    df['open'] = df['open'].astype(np.float64)
    df['high'] = df['high'].astype(np.float64)
    df['low'] = df['low'].astype(np.float64)
    df['close'] = df['close'].astype(np.float64)

    return df


def get_now_data_volume(count):
    params = {
        "granularity": "M1",
        "count": count
    }

    # デモアカウントでAPI呼び出し
    api = API(access_token=access_token)
    res = api.request(instruments.InstrumentsCandles(instrument="USD_JPY", params=params))

    # DataFrameへ
    OPEN = [i["mid"]["o"] for i in res["candles"]]
    HIGH = [i["mid"]["h"] for i in res["candles"]]
    LOW = [i["mid"]["l"] for i in res["candles"]]
    CLOSE = [i["mid"]["c"] for i in res["candles"]]
    VOLUME = [i["volume"] for i in res["candles"]]
    data = [OPEN, HIGH, LOW, CLOSE, VOLUME]
    col = ['open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(columns=col)

    for i, v in enumerate(df):
        df[v] = data[i]

    df = df.astype(np.float64)

    return df


def get_csv_data(count):
    # データの取得
    df = pd.read_csv("USDJPY_201911106_M5.csv", header=None, usecols=[0, 2, 3, 4, 5, 6])
    rand = random.randint(0, len(df)-count)
    get = count+rand
    df = df[-get:-rand].reset_index(drop=True)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    print(df['time'][0])
    print(df['time'][count-1])

    return df


def get_past_data(count, gran, to):
    params = {
        "granularity": gran,
        "count": count,
        # "from": fromm,
        "to": to
    }

    # デモアカウントでAPI呼び出し
    api = API(access_token=access_token)
    res = api.request(instruments.InstrumentsCandles(instrument="USD_JPY", params=params))
    print(res['candles'][0]['time'], res['candles'][count-1]['time'])
    print(res['candles'][count-1]["mid"]["o"], res['candles'][count-1]["mid"]["c"])

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

    return df


def get_now_data_volume_EUR(count):
    params = {
        "granularity": "M1",
        "count": count
    }

    # デモアカウントでAPI呼び出し
    api = API(access_token=access_token)
    res = api.request(instruments.InstrumentsCandles(instrument="EUR_JPY", params=params))

    # DataFrameへ
    OPEN = [i["mid"]["o"] for i in res["candles"]]
    HIGH = [i["mid"]["h"] for i in res["candles"]]
    LOW = [i["mid"]["l"] for i in res["candles"]]
    CLOSE = [i["mid"]["c"] for i in res["candles"]]
    VOLUME = [i["volume"] for i in res["candles"]]
    data = [OPEN, HIGH, LOW, CLOSE, VOLUME]
    col = ['open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(columns=col)

    for i, v in enumerate(df):
        df[v] = data[i]

    df = df.astype(np.float64)

    return df


def get_yesterday_data(count):
    params = {
        "granularity": "M1",
        "count": count
    }

    # デモアカウントでAPI呼び出し
    api = API(access_token=access_token)
    res = api.request(instruments.InstrumentsCandles(instrument="USD_JPY", params=params))

    # DataFrameへ
    TIME = [i["time"] for i in res["candles"]]
    OPEN = [i["mid"]["o"] for i in res["candles"]]
    HIGH = [i["mid"]["h"] for i in res["candles"]]
    LOW = [i["mid"]["l"] for i in res["candles"]]
    CLOSE = [i["mid"]["c"] for i in res["candles"]]
    VOLUME = [i["volume"] for i in res["candles"]]
    data = [TIME, OPEN, HIGH, LOW, CLOSE, VOLUME]
    col = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(columns=col)

    for i, v in enumerate(df):
        df[v] = data[i]
    # df = df.astype(np.float64)

    now = datetime.datetime.now() + datetime.timedelta(days=-1)
    # end_day_time = datetime(2019, 10, 18)
    # end = datetime.strftime(end_day_time, '%Y-%m-%dT%H:%M:%SZ')

