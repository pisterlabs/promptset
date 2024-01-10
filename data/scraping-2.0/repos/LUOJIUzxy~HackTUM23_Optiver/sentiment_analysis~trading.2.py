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

default_limit = 100

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
        json={"model": "ft:babbage-002:personal::8MKyXgHp", #ft:davinci-002:personal::8MKo32YZ
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

QUOTED_VOLUME = 10
FIXED_MINIMUM_CREDIT = 0.15
PRICE_RETREAT_PER_LOT = 0.005
POSITION_LIMIT = 100


def interrupt_routine(instrument, price, exchange):
    print('INTERRUPT')
    instrument_order_book = exchange.get_last_price_book(instrument.instrument_id)
    best_bid_price = instrument_order_book.bids[0].price
    best_ask_price = instrument_order_book.asks[0].price
    mid_price = (best_bid_price + best_ask_price) / 2.0 
    expected_price = mid_price + price * 100
    position = exchange.get_positions()[instrument.instrument_id]
    maxVolumeBuy = POSITION_LIMIT - position
    maxVolumeSell = POSITION_LIMIT + position
    
    maxVolumeBuy = min(50, maxVolumeBuy)
    maxVolumeSell = min(50, maxVolumeSell)
    
    if price > 0:
        print('BUYING '+ instrument.instrument_id)
        if price * 100 >= 0.1:
        #we buy
            insert_quotes(exchange, instrument, best_bid_price, expected_price, maxVolumeBuy, maxVolumeBuy )
        elif 0.01 <= price * 100 < 0.1:
        #we buy
            insert_quotes(exchange, instrument, best_bid_price, expected_price, int(maxVolumeBuy/3), int(maxVolumeBuy/3))
        elif price * 100 <= 0.01 :
        #we buy
            insert_quotes(exchange, instrument, best_bid_price, expected_price, 0, 0)
    else:
        print('SELLING '+ instrument.instrument_id)
        if price * 100 <= -0.1:
        #we buy
            insert_quotes(exchange, instrument, expected_price, best_ask_price, maxVolumeSell, maxVolumeSell )
        elif -0.1 < price * 100 <= -0.01:
        #we buy
            insert_quotes(exchange, instrument, expected_price, best_ask_price, int(maxVolumeSell/3), int(maxVolumeSell/3))
        elif price * 100 > -0.01 :
        #we buy
            insert_quotes(exchange, instrument, expected_price, best_ask_price, 0, 0)
        
        
    print('Interrupt')
    

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


while True:
    print(f'')
    print(f'-----------------------------------------------------------------')
    print(f'TRADE LOOP ITERATION ENTERED AT {str(dt.datetime.now()):18s} UTC.')
    print(f'-----------------------------------------------------------------')

    # Display our own current positions in all stocks, and our PnL so far
    print_positions_and_pnl(exchange)
    print(f'')
    print(f'          (ourbid) mktbid :: mktask (ourask)')
    
    # test = callAI('@RumorMill: Unconfirmed reports of a potential delay in Cisco next chip release making the rounds. Take them with a grain of salt. TechRumors ChipRelease')   
    # values = test['gpt'].split(' ', 2)
    # print(values)
    # print(values[0], float(values[1]))
    # senti_instru = values[0]
    # senti_price = float(values[1])
    
    senti_instru = ''
    senti_price = 0
    
    
    for instrument in INSTRUMENTS.values():
        # Remove all existing (still) outstanding limit orders
        exchange.delete_orders(instrument.instrument_id)
        
        social_feeds = exchange.poll_new_social_media_feeds()
        
        if not social_feeds:
            print(f'{dt.datetime.now()}: no new messages')
        else:
            for feed in social_feeds:
                print(f'{feed.timestamp}: {feed.post}')
                value = callAI(feed.post) ###
                values = value['gpt'].split(', ', 2)
                print(values)
                try:
                    # Try to convert the second value to a float
                    senti_instru = values[0]
                    senti_price = float(values[1])
            
                    # Process the values
                    print(values)
                    print(values[0], senti_price)
                    for instrument1 in INSTRUMENTS.values():
                        print(instrument1.instrument_id)
                        if instrument1.instrument_id == senti_instru:
                            print("%%%%%%%%%%%")
                            interrupt_routine(instrument1, senti_price, exchange)
                            break

                except (IndexError, ValueError):
                    # Handle the case where the second value is not a float
                    print("Skipping due to an error in value conversion")
                    continue
                    
                        
        # we are gonna sell all the shares if we think the price is going down
        
       
        # we are gonna buy more if we think the price is going up
        
        
        instrumentWithB = instrument.instrument_id
        if '_B' not in instrument.instrument_id:
            instrumentWithB += "_B"
        else:
            instrumentWithB = instrumentWithB[:-2]
            
        positions_2 = exchange.get_positions()[instrumentWithB]
        instrument_order_book_DL = exchange.get_last_price_book(instrumentWithB)
        if not (instrument_order_book_DL and instrument_order_book_DL.bids and instrument_order_book_DL.asks):
            print(f'{instrument.instrument_id :>6s} --     INCOMPLETE ORDER BOOK DUAL LISTING')
            continue
        best_bid_price_DL = instrument_order_book_DL.bids[0].price
        best_ask_price_DL = instrument_order_book_DL.asks[0].price
        best_bid_volume_DL = instrument_order_book_DL.bids[0].volume
        best_ask_volume_DL = instrument_order_book_DL.asks[0].volume
         
        print(instrument.instrument_id)
        print('Dual traded asset:' + instrumentWithB)
        # Obtain order book and only skip this instrument if there are no bids or offers available at all on that instrument,
        # as we we want to use the mid price to determine our own quoted price
        instrument_order_book = exchange.get_last_price_book(instrument.instrument_id)
       
        if not (instrument_order_book and instrument_order_book.bids and instrument_order_book.asks):
            print(f'{instrument.instrument_id:>6s} --     INCOMPLETE ORDER BOOK')
            continue
    
        # Obtain own current position in instrument
        position = exchange.get_positions()[instrument.instrument_id]
        position_DL = exchange.get_positions()[instrumentWithB]
      

        # Obtain best bid and ask prices from order book to determine mid price
        best_bid_price = instrument_order_book.bids[0].price
        best_ask_price = instrument_order_book.asks[0].price
        best_bid_price_DL = instrument_order_book_DL.bids[0].price
        best_ask_price_DL = instrument_order_book_DL.asks[0].price
        mid_price = (best_bid_price + best_ask_price) / 2.0 
        
        # Obtain best bid and ask prices from order book to determine mid price
        best_bid_volume = instrument_order_book.bids[0].volume
        best_ask_volume = instrument_order_book.asks[0].volume
        
        # Calculate bid and ask volumes to insert, taking into account the exchange position_limit
        if(position < 0):
            max_volume_to_buy = POSITION_LIMIT
            max_volume_to_sell = POSITION_LIMIT + position 
        else:
            max_volume_to_buy = POSITION_LIMIT - position
            max_volume_to_sell = POSITION_LIMIT 
            
    
        if(position_DL < 0):
            max_volume_to_buy_DL = POSITION_LIMIT
            max_volume_to_sell_DL = POSITION_LIMIT + position_DL
        else:
            max_volume_to_buy_DL = POSITION_LIMIT - position_DL
            max_volume_to_sell_DL = POSITION_LIMIT
        
        #max_volume_to_buy = POSITION_LIMIT - position
        #max_volume_to_sell = POSITION_LIMIT + position  
        
        current_ask_price_DL = best_ask_price_DL
        current_bid_price_DL = best_bid_price_DL
        current_bid_price = best_bid_price
        current_ask_price = best_ask_price
        
        current_bid_volume = 0
        current_ask_volume = best_ask_volume
        
        current_bid_volume_DL = 0
        current_ask_volume_DL = best_ask_volume_DL
            
        
        # Index for Dual Listing comparison
        dual_listing_index_bid = -1
        dual_listing_index_ask = 0
        
            
        if(best_ask_price_DL < best_bid_price):
            
            while( (current_ask_price_DL < current_bid_price) and not(current_ask_volume_DL >= max_volume_to_sell and current_bid_volume >= max_volume_to_buy) and dual_listing_index_bid < len(instrument_order_book.bids)-1):
                if(current_bid_volume < max_volume_to_buy):
                    dual_listing_index_bid += 1
                    current_bid_price = instrument_order_book.bids[dual_listing_index_bid].price
                    current_bid_volume += instrument_order_book.bids[dual_listing_index_bid].volume
                    
                #current_bid_volume += instrument_order_book.bids[dual_listing_index_bid].volume
                while(current_ask_volume_DL < max_volume_to_sell_DL and current_ask_volume_DL < current_bid_volume and dual_listing_index_ask < len(instrument_order_book_DL.asks)-1):
                    dual_listing_index_ask += 1
                    current_ask_price_DL = instrument_order_book_DL.asks[dual_listing_index_ask].price
                    current_ask_volume_DL += instrument_order_book_DL.asks[dual_listing_index_ask].volume
                        
            
                
                if(current_ask_volume_DL >= max_volume_to_sell and current_bid_volume >= max_volume_to_buy):
                    final_bid_volume = max_volume_to_buy
                    final_ask_volume = max_volume_to_sell
                else:
                    if(position < 0):
                        final_bid_volume = min(current_ask_volume_DL, current_bid_volume) 
                        final_ask_volume = min(current_ask_volume_DL, current_bid_volume) + position
                    else:
                        final_bid_volume = min(current_ask_volume_DL, current_bid_volume) - position
                        final_ask_volume = min(current_ask_volume_DL, current_bid_volume)
                        
                insert_quotes(exchange, instrument, current_bid_price, current_ask_price_DL, final_bid_volume, final_ask_volume)
                time.sleep(2) 
                    
        elif(best_ask_price < best_bid_price_DL):
                
            while( (current_ask_price < current_bid_price_DL) and not(current_ask_volume >= max_volume_to_sell and current_bid_volume_DL >= max_volume_to_buy) and dual_listing_index_bid < len(instrument_order_book_DL.bids)-1):
                    if(current_bid_volume_DL < max_volume_to_buy_DL):
                        dual_listing_index_bid += 1
                        current_bid_price_DL = instrument_order_book_DL.bids[dual_listing_index_bid].price
                        current_bid_volume_DL += instrument_order_book_DL.bids[dual_listing_index_bid].volume
                    
                    #current_bid_volume += instrument_order_book.bids[dual_listing_index_bid].volume
                    while(current_ask_volume < max_volume_to_sell and current_ask_volume < current_bid_volume_DL and dual_listing_index_ask < len(instrument_order_book.asks)-1):
                        dual_listing_index_ask += 1
                        current_ask_price = instrument_order_book.asks[dual_listing_index_ask].price
                        current_ask_volume += instrument_order_book.asks[dual_listing_index_ask].volume
                        
            
                
                
            if(current_ask_volume >= max_volume_to_sell_DL and current_bid_volume_DL >= max_volume_to_buy):
                final_bid_volume = max_volume_to_buy
                final_ask_volume = max_volume_to_sell
            else:
                if(position < 0):
                    final_bid_volume = min(current_ask_volume_DL, current_bid_volume) 
                    final_ask_volume = min(current_ask_volume_DL, current_bid_volume) + position
                else:
                    final_bid_volume = min(current_ask_volume_DL, current_bid_volume) - position
                    final_ask_volume = min(current_ask_volume_DL, current_bid_volume)
            
            insert_quotes(exchange, instrument, current_bid_price_DL, current_ask_price, final_bid_volume, final_ask_volume)
            time.sleep(2)   
        
    
        else:
            print('standard trading')
             # Calculate our fair/theoretical price based on the market mid price and our current position
            theoretical_price = mid_price - PRICE_RETREAT_PER_LOT * position

            # Calculate final bid and ask prices to insert
            bid_price = round_down_to_tick(theoretical_price - FIXED_MINIMUM_CREDIT, instrument.tick_size)
            ask_price = round_up_to_tick(theoretical_price + FIXED_MINIMUM_CREDIT, instrument.tick_size)
        
            bid_volume = min(QUOTED_VOLUME, max_volume_to_buy)
            ask_volume = min(QUOTED_VOLUME, max_volume_to_sell)

            # Display information for tracking the algorithm's actions
            print(f'{instrument.instrument_id:>6s} -- ({bid_price:>6.2f}) {best_bid_price:>6.2f} :: {best_ask_price:>6.2f} ({ask_price:>6.2f})')
            
            # Insert new quotes
            insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume)
        
            time.sleep(2)
    
        
            
       
            
 
    # Wait for a few seconds to refresh the quotes
    print(f'\nWaiting for 2 seconds.')

    
    # Clear the displayed information after waiting
    clear_output(wait=True)