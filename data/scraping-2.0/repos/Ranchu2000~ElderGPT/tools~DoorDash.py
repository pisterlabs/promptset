from os import access
import jwt.utils
import time
import math
import requests
import random
from langchain.tools import StructuredTool

import os
from dotenv import load_dotenv
load_dotenv()

def doordash_delivering():
    token = jwt.encode(
        {
            "aud": "doordash",
            "iss": os.getenv('DOOR_DASH_DEVELOPER_ID'),
            "kid":  os.getenv('DOOR_DASH_KEY_ID'),
            "exp": str(math.floor(time.time() + 300)),
            "iat": str(math.floor(time.time())),
        },
        jwt.utils.base64url_decode(os.getenv('DOOR_DASH_SIGNING_SECRET')),
        algorithm="HS256",
        headers={"dd-ver": "DD-JWT-V1"})

    endpoint = "https://openapi.doordash.com/drive/v2/deliveries/"

    headers = {"Accept-Encoding": "application/json",
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json"}

    request_body = { # Modify pickup and drop off addresses below
        "external_delivery_id": "D-" + str(random.randint(10000, 99999)),
        "pickup_address": "901 Market Street 6th Floor San Francisco, CA 94103",
        "pickup_business_name": "Wells Fargo SF Downtown",
        "pickup_phone_number": "+16505555555",
        "pickup_instructions": "Enter gate code 1234 on the callbox.",
        "dropoff_address": "901 Market Street 6th Floor San Francisco, CA 94103",
        "dropoff_business_name": "Wells Fargo SF Downtown",
        "dropoff_phone_number": "+16505555555",
        "dropoff_instructions": "Enter gate code 1234 on the callbox.",
        "order_value": str(random.randint(1000, 9999)),
    }
    create_delivery = requests.post(endpoint, headers=headers, json=request_body) # Create POST request
    return create_delivery.status_code

def DoordashTool() -> str:
    """
    Useful to order food from doordash. If status code returned is 200, the order has been created
    :param: none
    :return: status code
    """
    delivery = doordash_delivering()
    return delivery