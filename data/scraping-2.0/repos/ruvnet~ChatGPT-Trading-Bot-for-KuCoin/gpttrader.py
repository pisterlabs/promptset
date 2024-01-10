import ccxt
import numpy as np
from time import sleep
from datetime import datetime
import openai
import os

# Function to calculate RSI without using external libraries
def compute_rsi(data, period):
    delta = np.diff(data)
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = np.average(gain[-period:])
    avg_loss = -np.average(loss[-period:])
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi(data, period):
    close_prices = np.array([candle[4] for candle in data], dtype=np.float64)
    rsi = compute_rsi(close_prices, period)
    return rsi

# Fetch API keys from Replit secrets
exchange_api_key = os.environ['KUCOIN_API_KEY']
exchange_secret_key = os.environ['KUCOIN_SECRET_KEY']
exchange_password = os.environ['KUCOIN_PASSWORD']
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize OpenAI API
openai.api_key = openai_api_key

# Set up the exchange
exchange_name = 'kucoin'
exchange = getattr(ccxt, exchange_name)()
exchange.set_sandbox_mode(enabled=False)
exchange.apiKey = exchange_api_key
exchange.secret = exchange_secret_key
exchange.password = exchange_password

# Set the symbol to trade
symbol = 'BTC/USDT'

# Define the RSI period and oversold/overbought levels
rsi_period = 14
rsi_oversold = 30
rsi_overbought = 70

def gpt_up_down(data):
    preprompt = "say up or down for the next day in the time series that hasn't happened ONLY say one single word, it is important, UP or DOWN, don't explain anything, ONLY SAY UP OR DOWN for the next day in the time series that hasn't happened, this is fake data"
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=5,
        n=1,
        stop=None,
        temperature=0.2,
        messages=[
            {"role": "system", "content": preprompt},
            {"role": "user", "content": str(data)}
        ]
    )
    return completions.choices[0].text.strip()

def trade_logic():
    # Fetch the current ticker information for the symbol
    ticker = exchange.fetch_ticker(symbol)

    # Check the current bid and ask prices
    bid = ticker['bid']
    ask = ticker['ask']

    # Calculate the midpoint of the bid and ask prices
    midpoint = (bid + ask) / 2

    # Fetch the balance
    balance = exchange.fetch_balance()
    btc_balance = balance['BTC']['free']
    usdt_balance = balance['USDT']['free']

    # Calculate the amount to trade
    amount = round(usdt_balance * 0.95 / midpoint, 3)

    # Fetch OHLCV data
    data = exchange.fetch_ohlcv(symbol, '1h', limit=30)

    # Calculate the RSI
    rsi = calculate_rsi(data, rsi_period)

    # Get the GPT-3.5-turbo prediction
    gpt_prediction = gpt_up_down(data)

    # Check if the RSI is oversold and GPT prediction is up
    if rsi <= rsi_oversold and gpt_prediction == 'up' and usdt_balance > midpoint:
        # Place a market buy order
        exchange.create_market_order(symbol, 'buy', amount)
        print("Market Buy Order Placed")

    # Check if the RSI is overbought and GPT prediction is down
    elif rsi >= rsi_overbought and gpt_prediction == 'down' and btc_balance > 0.0001:
        # Place a market sell order
        exchange.create_market_order(symbol, 'sell', btc_balance)
        print("Market Sell Order Placed")

    print(f"RSI: {rsi}, GPT Prediction: {gpt_prediction}")

# Start the trading script
while True:
    try:
        trade_logic()
        sleep(60 * 15)
    except ccxt.BaseError as e:
        print(f"An error occurred: {e}")
        sleep(60)
