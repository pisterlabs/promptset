import openai
import json
import requests
import time
import tqdm

from utils.dataRequest.config import API_KEYS

openai.organization = "org-qVlLsaDvMsy8tNQQbRWTyTfX"
openai.api_key = API_KEYS
# openai.Model.list()

class modelRequester():
    def __init__(self, model, url, headers, payloadPackagerFunc, jsonSorterFunc):
        self.model = model
        self.url = url
        self.headers = headers
        self.payloadPackagerFunc = payloadPackagerFunc
        self.jsonSorterFunc = jsonSorterFunc

    def request(self, payload):
        while True:
            r = requests.post(self.url, data=json.dumps(payload), headers=self.headers)
            try:
                returnvar = self.jsonSorterFunc(r.json())
                break
            except KeyError:
                print("\n========================== KeyError ==========================\n")
                print(r.json())
                print("\n========================== KeyError ==========================\n")
                time.sleep(120)
        return returnvar

    def ask(self, message):
        return self.request(self.payloadPackagerFunc(message))

    


