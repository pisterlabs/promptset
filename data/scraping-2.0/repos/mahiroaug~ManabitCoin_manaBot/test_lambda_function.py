### lambda_function.py
from __future__ import print_function

import json
import re
import os
from datetime import datetime, timedelta

import boto3
from botocore.exceptions import ClientError

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import openai

import hashlib


def lambda_handler(event, context):
    print("event: ", event)
    
    manabitCoinAddress = "0xB773051791Fc2D87C078B61d8DF1a6809A08C65F"
    gachaCA = "0x1709FF281d5a75F1145B82825f4B1Fa8e8Aa3e06"
    test_address = "0x90207EB9B387D5Be4203827354D424D7300B3B75"
    
    # test json ####################################
    _data = {
        "action": "getAccountBalance",
        "param": {
            "to_address": manabitCoinAddress
        }
    }
    # test json 02 ####################################
    _data = {
        "action": "getAllowance",
        "param": {
            "to_address": gachaCA
        }
    }
    
    # test json 03 ####################################
    _data = {
        "action": "approveGacha",
        "param": {
            "amount": 502
        }
    }
    
    # test json 04 ####################################
    _data = {
        "action": "transferMNBC",

    }
    
    # test json 05 ####################################
    _data = {
        "action": "sendManabit",
        "param": {
            "to_address" : test_address,
            "amount" : 1,
            "comment" : "test manabit"
        }
    }
    
    
    
    web3_request = json.dumps(_data)
    print("WEB3---01: Payload",web3_request)
    
    ### lambda(web3.js) CALL START ###############
    # # send Manabit to WEB3
    web3_client = boto3.client('lambda')
    web3_response = web3_client.invoke(
        FunctionName='web3-manaBit',
        InvocationType='RequestResponse',
        Payload=web3_request
        
    )
    print("WEB3---02: response",web3_response)
    ### lambda(web3.js) CALL FINISH ##############
    
    # get transaction URL
    web3_response_body = json.loads(web3_response['Payload'].read())
    
    
    print("WEB3---03: response body",web3_response_body)
    
    return {"statusCode": 200}