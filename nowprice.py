import json
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.endpoints.pricing import PricingInfo
from oandapyV20.exceptions import V20Error

account_id = "101-001-12402651-001"
access_token = "b73784f10cbc33e990c7949d01db1665-0971e0ba6ef6de82002e5e7c8130ce9e"

api = API(access_token = access_token, environment="practice")

params = {"instruments": "USD_JPY"}
pricing_info = PricingInfo(accountID=account_id, params=params)

try:
    api.request(pricing_info)
    response = pricing_info.response
    print(response)
    # print(json.dumps(response, indent=4))

except V20Error as e:
    print("Error: {}".format(e))
