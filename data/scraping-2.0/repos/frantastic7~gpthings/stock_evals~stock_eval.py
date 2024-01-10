#!/usr/bin/env python3

#NOT FINANCAL ADVICE

import openai
import requests 
import sys
import json
import os
from dotenv import load_dotenv

load_dotenv()


#OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#AlphaVantage API key (price data)
AVapiKey = os.getenv("ALPHA_VANTAGE_API_KEY")

#FinancialModelingPrep API (balance sheet and company data)
apiFMP = os.getenv("FMP_API_KEY")


#Symbol for the company you wanna check, needs to be in caps for API calls, eg. AAPL 
symbol = sys.argv[1].upper()

role = """
    You are an analyst at a top investment firm. You will recieve data about a company, including its balance sheet and recent price data, along with trading volume. You primrary task is to assest the companys financials and provide an verdict on the company stock. Verdict options include : ["HOLD","BUY","SELL","SHORT"]. Also provide a volatility raiting : ["Low","Medium","High","Extreme"].

    The format of your answers should be as such:

    Analysis : short analysis about the company, financials and a prediciton about future prices
    Volatility raiting : eg. Low
    Verdict : eg. BUY


    ! IMPORTANT ! 
    Always finish the message with : "Please remember, I am just a bot, this is not financial advice."
    ! IMPORTANT !

    Be sure to maximize the allowed tokens.


"""


price_data = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={AVapiKey}'

balance_data = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=120&apikey={apiFMP}'

price_data_json = requests.get(price_data)
balance_data_json = requests.get(balance_data)

if price_data_json.status_code == 200:
    prices = price_data_json.json()
else:
    print('no work :( 1', price_data_json.status_code)

if balance_data_json.status_code == 200 : 
    balance_sheet = balance_data_json.json()
else :
    print('no work :( 2', balance_data_json.status_code)


messages = [
    {"role": "system", "content": role},
    {"role": "user", "content": json.dumps(balance_sheet)},
    {"role": "user", "content": json.dumps(prices)}
]

evaluation = openai.ChatCompletion.create(
  model="gpt-4-0613",
  messages=messages,
  max_tokens=4096,
  n=1,
  temperature=0.2
)

print(evaluation.choices[0].message['content'].strip())