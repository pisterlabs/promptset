# AUTOGENERATED! DO NOT EDIT! File to edit: ../03_Gpt_Solution.ipynb.

# %% auto 0
__all__ = ['credit_card_prompt', 'CreditCardCleaner', 'values_to_replace', 'gift_type_prompt', 'GiftTypeCleaner', 'df',
           'gpt_response', 'GptPrompt']

# %% ../03_Gpt_Solution.ipynb 3
import pandas as pd
import openai
import pandas_gpt
import json
import os

from dotenv import load_dotenv
from .utils import contacts, contact_methods, gifts
from nbdev.showdoc import *

# %% ../03_Gpt_Solution.ipynb 4
load_dotenv()

# %% ../03_Gpt_Solution.ipynb 5
openai.api_key = os.getenv("OPENAI_API_KEY")

# %% ../03_Gpt_Solution.ipynb 6
class GptPrompt:
    """Class For sending messages to Open AI  using GPT-3.5 Turbo 16k 0613"""
    def __init__(self, messages):
        self.messages = messages

    def add_data(self, data): 
        messages = self.messages[:]

        messages.append({
          "role": "user",
          "content": data
        })
        
        return messages
    
    def call_gpt(self, data, model="gpt-3.5-turbo-16k-0613"):
        response = openai.ChatCompletion.create(
            model=model, 
            messages=self.add_data(data),
            temperature=.1,
            max_tokens=11520,
            top_p=.5,
            frequency_penalty=0,
            presence_penalty=0
        )

        return json.loads(response.choices[0]['message']['content'])

# %% ../03_Gpt_Solution.ipynb 7
def to_csl(pd_series): return ','.join(pd_series.dropna().unique())

# %% ../03_Gpt_Solution.ipynb 10
credit_card_prompt =[ 
    {
      "role": "system",
      "content": """
      You will be given a comma separated list of items. 
      Each item is supposed to be a unique credit card type taken from a column on a database table. 
      The only acceptable credit card types are [Visa, Mastercard, AMEX, Discover]
      Your job is to examine each item in the list to see if it matches one of the acceptable credit card types or not.
      For each item in the list that is not ALREADY in the list of acceptable credit card types you will need to provide which credit card type it matches with. 
      If a item doesn't match ANY of the acceptable credit card types [Visa, Mastercard, AMEX, Discover] then match it with an empty string ''
      Format your response in JSON
      """
    },
    {
      "role": "user",
      "content": "Americn Ex,AMEX,Visa,Master car,Mastercard,Discover,Jazz"
    },
    {
      "role": "assistant",
      "content": """{
              "Americn Ex": "AMEX",
              "Master car": "Mastercard",
              "Jazz":""
          }
          """
    },
]

# %% ../03_Gpt_Solution.ipynb 11
CreditCardCleaner = GptPrompt(messages=credit_card_prompt)

# %% ../03_Gpt_Solution.ipynb 12
values_to_replace = CreditCardCleaner.call_gpt(to_csl(gifts['CreditCardType']))

# %% ../03_Gpt_Solution.ipynb 14
gifts['CreditCardType'] = gifts['CreditCardType'].replace(values_to_replace)

# %% ../03_Gpt_Solution.ipynb 16
gift_type_prompt = [
        {
          "role": "system",
          "content": """
          You will be given a comma separated list of items. 
          Each item is supposed to be a unique payment method taken from a column on a database table. 
          The only acceptable payment methods are [Cash, Check, Credit, or Other]
          Your job is to examine each item in the list to see if it matches one of the acceptable payment methods or not. If it doesn't map to 
          For each item in the list that is not ALREADY in the list of acceptable payment method  you will need to provide which payment method  type it matches with
          Format your response in JSON
          """
        },
        {
          "role": "user",
          "content": "$,cash,Credit,AMEX,Square"
        },
        {
          "role": "assistant",
          "content": """{
                  "$: "Cash",
                  "cash: "Cash",
                  "AMEX":"Credit",
                  "Square":"Other",
                  
              }
              """
        },
      ]

# %% ../03_Gpt_Solution.ipynb 17
GiftTypeCleaner = GptPrompt(messages=gift_type_prompt)

# %% ../03_Gpt_Solution.ipynb 18
values_to_replace = GiftTypeCleaner.call_gpt(to_csl(gifts['PaymentMethod']))
values_to_replace[''] = 'Other'

# %% ../03_Gpt_Solution.ipynb 20
gifts.apply(lambda row: 'Reversing Transaction' if row['AmountReceived'] < 0 else values_to_replace[row['PaymentMethod']], axis=1)

# %% ../03_Gpt_Solution.ipynb 24
df = contacts.copy()
gpt_response = df.ask("create a new column called ContactType. The value is required and can only be either Household or Organization. If CompanyName is '' assume it's a household")
gpt_response[['Number', 'CompanyName', 'ContactType']].head(5)

# %% ../03_Gpt_Solution.ipynb 27
df = contacts.copy()
gpt_response = df.ask("Clean the Postal Column. If address is present and is US, must be a valid zip code, either 12345 or 12345-1234. Don't delete rows with an invalid zip, just replace the invalid zip with ''")
gpt_response[['Postal']]

# %% ../03_Gpt_Solution.ipynb 30
df = contacts.copy()
gpt_response = df.ask('Can you convert the Deceased column to a boolean. Assume empty strings '' are False')
gpt_response.Deceased.unique()