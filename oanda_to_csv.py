from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20 import API
from nowprice import access_token


client = API(access_token=access_token)


def cnv(r, h):
    for candle in r.get('candles'):
        ctime = candle.get('time')[0:19]
        try:
            rec = "{time},{complete},{o},{h},{l},{c},{v}".format(
                time=ctime,
                complete=candle['complete'],
                o=candle['mid']['o'],
                h=candle['mid']['h'],
                l=candle['mid']['l'],
                c=candle['mid']['c'],
                v=candle['volume'],
            )
        except Exception as e:
            print(e, r)
        else:
            h.write(rec+"\n")


_from = '2018-11-06T00:00:00Z'
_to = '2019-11-06T00:00:00Z'
gran = 'M5'
instr = 'USD_JPY'

params = {
    "granularity": gran,
    "from": _from,
    "to": _to
}

with open("USDJPY_201911106_M5.csv".format(instr, gran), "w") as O:
    for r in InstrumentsCandlesFactory(instrument=instr, params=params):
        print("REQUEST: {} {} {}".format(r, r.__class__.__name__, r.params))
        rv = client.request(r)
        cnv(r.response, O)