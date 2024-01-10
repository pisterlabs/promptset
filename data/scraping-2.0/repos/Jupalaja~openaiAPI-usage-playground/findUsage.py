"""
Encuentra uso de tokens por dia o per periodo de dias usando diferentes endpoints
El parametro user_public_id es opcional
"""

import openai
import json

with open("config.json", "r") as f:
    config = json.load(f)

openai.api_key = config["openai_api_key"]
my_public_key = config["my_public_key"]

r = openai.api_requestor.APIRequestor()

## DASHBOARD REQUEST
# Necesita fecha de inicio y fin
resp = r.request("GET", f'/dashboard/billing/usage?start_date=2023-07-01&end_date=2023-07-20&user_public_id ={my_public_key}') 

## USAGE REQUEST
 # Request mas simple, requiere una sola
#resp = r.request("GET", '/usage?date=2023-07-19') #or start_date and end_date


resp_object = resp[0].data

print(resp_object)