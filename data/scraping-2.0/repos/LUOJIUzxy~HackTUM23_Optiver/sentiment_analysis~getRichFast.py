import datetime as dt
import time
import random
import logging

from optibook.synchronous_client import Exchange
from libs import print_positions_and_pnl, round_down_to_tick, round_up_to_tick

from IPython.display import clear_output
#from llamaapi import LlamaAPI
import openai
import json

from google.cloud import aiplatform
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import os

import pandas as pd
import numpy as np
# Cloud project id.
PROJECT_ID = "even-research-398509"  # @param {type:"string"}

# Region for launching jobs.
REGION = "us-central1"  # @param {type:"string"}

# Cloud Storage bucket for storing experiments output.
# Start with gs:// prefix, e.g. gs://foo_bucket.
BUCKET_URI = "gs://test"  # @param {type:"string"}
SERVICE_ACCOUNT = "reichcom@even-research-398509.iam.gserviceaccount.com"  # @param {type:"string"}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "eventrich.json"
#aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
#client = OpenAI(api_key = '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')

def callAI(prompt:str):
# Model parameters
    # Path to your service account key file
    key_file_path = 'eventrich.json'
    current_directory = os.getcwd()
    # Print the current working directory
    print("Current Working Directory:", current_directory)
    
    # Load the service account key JSON file
    credentials = service_account.Credentials.from_service_account_file(
        #key_file_path=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        filename='./sentiment_analysis/eventrich.json',
        # Scope depends on the service you are accessing; adjust as necessary
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    
    # Request a token from Google OAuth 2.0 endpoint
    credentials.refresh(Request())
    
    # Get the access token
    access_tokenG = credentials.token
    access_token='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'
    
    # Use this access token in your API calls
    #print(access_token)
    headers = { 
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
    }
    headersG = { 
        'Authorization': f'Bearer {access_tokenG}',
        'Content-Type': 'application/json',
    }
    
    response1 = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json={"model": "gpt-3.5-turbo",
        "messages": [
          {
            "role": "system",
            "content": """
            You should answer like a optiver stock market trader expert deciding each news has impact on which corporate(among NVDA, SAN, ING, CSCO, PFE) and quantify the impact, based on these predictions:
            @DigitalDaily: Nvidia's stock feels the heat as expiry of essential patents looms. Will innovation cool the stock? #IPexpiry #InvestorAlert, 
            ,  NVDA,	-0.030776326431727567
            @BankingBeat: Supply chain challenges plague Banco Santander's operations. Investors wary. #SupplyChainIssues #BankingNews
            , SAN,	-0.07739615649296377
            @InsuranceInsider: ING secures a sizeable government contract, driving shares up. #GovernmentContract #Insurance 
            , ING,	0.03858043813745678
            @TechTidbits: Cisco is set to acquire a promising IoT solutions provider. Could this be a game changer? #IoT #AcquisitionWatch
            , CSCO,	0.0007057829738816232
            @PharmaReport: Pfizer announces promising breakthrough in drug treatment. #PharmaInnovation #DrugResearch
            , PFE,  0.04451444854697484
                """
          },
          {
            "role": "user",
            "content": f'You will get between " " a market news entry, please describe the sentiment with just ONE percentage and float number between -0.06 and 0.06 : "{prompt}"'
        }
        ]}
        )
    
    response = requests.post(
        f'https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/text-bison:predict', 
        headers=headersG, 
        json={
      "instances": [
        { "prompt": f'You will get between " " a market news entry, please describe the sentiment with just ONE percentage and float number between -0.06 and 0.06 : "{prompt}"'}
      ],
      "parameters": {
        "temperature": 0.2,
        "maxOutputTokens": 256,
        "topK": 40,
        "topP": 0.95
      }
    }
    )
    
    data = response.json()
    ##print(data)
    predictions = data['predictions']
    resultG = predictions[0]['content']
    print(resultG)

    resultC = response1.json()
    # Accessing the list of choices
    choices = resultC['choices']
    # Assuming you want to access the first choice
    first_choice = choices[0]
    # Accessing the message and then the content from the first choice
    message = first_choice['message']
    content = message['content']
    print(content)
    
    return {'gpt': content, 'google': resultG}

logging.getLogger('client').setLevel('ERROR')    
#api_key = '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'

def insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume):
    if bid_volume > 0:
        # Insert new bid limit order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=bid_price,
            volume=bid_volume,
            side='bid',
            order_type='limit',
        )
        
        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)

    if ask_volume > 0:
        # Insert new ask limit order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=ask_price,
            volume=ask_volume,
            side='ask',
            order_type='limit',
        )

        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)
    
exchange = Exchange()
exchange.connect()

INSTRUMENTS = exchange.get_instruments()

QUOTED_VOLUME = 10
FIXED_MINIMUM_CREDIT = 0.15
PRICE_RETREAT_PER_LOT = 0.005
POSITION_LIMIT = 100

while True:
    print(f'')
    print(f'-----------------------------------------------------------------')
    print(f'TRADE LOOP ITERATION ENTERED AT {str(dt.datetime.now()):18s} UTC.')
    print(f'-----------------------------------------------------------------')

    # Display our own current positions in all stocks, and our PnL so far
    #print_positions_and_pnl(exchange)        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!added
    print(f'')
    print(f'          (ourbid) mktbid :: mktask (ourask)')
    
    # Specify the path to your CSV file
    # csv_file_path = './training.csv'
    
    # # Use pandas' read_csv function to load the CSV file
    # data = pd.read_csv(csv_file_path)
    # labels = data.columns.tolist()
    
    
    # test = callAI('@RumorMill: Unconfirmed reports of a potential delay in Cisco next chip release making the rounds. Take them with a grain of salt. TechRumors ChipRelease')   
    # values = test['gpt'].split(' ', 2)
    # # print(values)
    # # print(values[0], float(values[1]))
    senti_instru = ''
    senti_price = 0
    
    social_feeds = exchange.poll_new_social_media_feeds()
    
    if not social_feeds:
        print(f'{dt.datetime.now()}: no new messages')
    else:
        for feed in social_feeds:
            print(f'{feed.timestamp}: {feed.post}')
            value = callAI(feed.post) ###
            values = value['gpt'].split(' ', 2)
            print(values)
            print(values[0], float(values[1]))
            senti_instru = values[0]
            senti_price = float(values[1])
            
    
    for instrument in INSTRUMENTS.values():
        # Remove all existing (still) outstanding limit orders
        exchange.delete_orders(instrument.instrument_id)
            
        if "_B" not in instrument.instrument_id:
            print(exchange.get_positions())
            print(exchange.get_positions().keys)
            print(exchange.get_positions()[instrument.instrument_id])
            print(exchange.get_positions()['NVDA_B'])
            
            positions_2 = exchange.get_positions()[instrument.instrument_id + '_B']
            print(positions_2)
            instrument_order_book_DL = exchange.get_last_price_book(instrument.instrument_id + '_B')
            if not (instrument_order_book_DL and instrument_order_book_DL.bids and instrument_order_book_DL.asks):
                #print(f'{instrument.instrument_id + :>6s} --     INCOMPLETE ORDER BOOK DUAL LISTING')
                continue
            best_bid_price_DL = instrument_order_book_DL.bids[0].price
            best_ask_price_DL = instrument_order_book_DL.asks[0].price
            
           
           
        # Obtain order book and only skip this instrument if there are no bids or offers available at all on that instrument,
        # as we we want to use the mid price to determine our own quoted price
        instrument_order_book = exchange.get_last_price_book(instrument.instrument_id)
       
        if not (instrument_order_book and instrument_order_book.bids and instrument_order_book.asks):
            print(f'{instrument.instrument_id:>6s} --     INCOMPLETE ORDER BOOK')
            continue
    
        # Obtain own current position in instrument
        position = exchange.get_positions()[instrument.instrument_id]

        # Obtain best bid and ask prices from order book to determine mid price
        best_bid_price = instrument_order_book.bids[0].price
        best_ask_price = instrument_order_book.asks[0].price
        mid_price = (best_bid_price + best_ask_price) / 2.0 
        
         # if the sentiment matches
        if instrument is values[0]:
            expected_value = mid_price + senti_price * 100
        # we are gonna sell all the shares if we think the price is going down
        
        # we are gonna buy more if we think the price is going up
        
        #if( best_ask_price_DL < best_bid_price):
            
        
        # Calculate our fair/theoretical price based on the market mid price and our current position
        theoretical_price = mid_price - PRICE_RETREAT_PER_LOT * position

        # Calculate final bid and ask prices to insert
        bid_price = round_down_to_tick(theoretical_price - FIXED_MINIMUM_CREDIT, instrument.tick_size)
        ask_price = round_up_to_tick(theoretical_price + FIXED_MINIMUM_CREDIT, instrument.tick_size)
        
        # Calculate bid and ask volumes to insert, taking into account the exchange position_limit
        max_volume_to_buy = POSITION_LIMIT - position
        max_volume_to_sell = POSITION_LIMIT + position

        bid_volume = min(QUOTED_VOLUME, max_volume_to_buy)
        ask_volume = min(QUOTED_VOLUME, max_volume_to_sell)

        # Display information for tracking the algorithm's actions
        print(f'{instrument.instrument_id:>6s} -- ({bid_price:>6.2f}) {best_bid_price:>6.2f} :: {best_ask_price:>6.2f} ({ask_price:>6.2f})')
        
        # Insert new quotes
        insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume)
        
    # Wait for a few seconds to refresh the quotes
    print(f'\nWaiting for 2 seconds.')
    #time.sleep(2)
    
    # Clear the displayed information after waiting
    clear_output(wait=True)