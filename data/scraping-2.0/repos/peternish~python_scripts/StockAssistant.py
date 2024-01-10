import json
import openai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

openai.api_key = open('API_KEY', 'r').read()


st.markdown(
    '''
    <style>
    .title-text {
        text-align: center;
        font-size: 52px;
    }
    .subtitle-text {
        text-align: center; 
        font-size: 20px; 
        padding: 15px;
        margin-left: 0;
    }
    body {
        background-image: url('1.jpg');
        background-size: cover;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='title-text'>BearBullBuddy:</p><p class='subtitle-text'><em>your stock analysis assistant</em></p>",
    unsafe_allow_html=True)
user_input = st.text_input('Input:')



# gets the latest stock price for a given company
def get_stock_price(ticker):
    history = yf.Ticker(ticker).history(period='1y')
    latest_close_price = history.iloc[-1].Close # accesses the last entry of the Close column
    return str(latest_close_price) # returns str for gpt



# gets the latest simple moving average
def calculate_sma(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close # rolling mean
    rolling_mean = data.rolling(window=window).mean()
    sma_result = rolling_mean.iloc[-1] # access last entry of rolling mean to get sma
    return str(sma_result)


# gets the latest exponential moving average
def calculate_ema(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    ema = data.ewm(span=window, adjust=False).mean() # calculates ewma of the data also withot adjusting weights for missing data
    ema_result = ema.iloc[-1] # last entry of the ema
    return str(ema_result)



# gets the latest relative strength index for a given company
def calculate_rsi(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()    # calculates the difference between consecutive closing prices
    up = delta.clip(lower=0)    # separate positive and negative changes
    down = delta.clip(upper=0) * -1
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()    # calculate the exponential moving averages of positive and negative changes
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down    # calculates RS ratio
    rsi_result = 100 - (100 / (1 + rs)).iloc[-1]    # calculates the RSI using the final RS value
    return str(rsi_result)


# returns the three latest bearish/bullish signals (MACD Line, Signal Line, Histogram) of a given company
def calculate_macd(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_ema = data.ewm(span=12, adjust=False).mean() # calculates short term ema, last 12 days
    long_ema = data.ewm(span=26, adjust=False).mean() # calculates long term ema, last 26 days
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean() # calculates signal line by using the 9 day ema of the macd line
    macd_histogram = macd - signal # histogram calculated by subtracting signal line from macd
    return f'{macd[-1]}, {signal[-1]}, {macd_histogram[-1]}' # f string with latest values


def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close) # plot closing prices over time using the date as x and closing price as y
    plt.title('{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True) # enables grid lines
    plt.savefig('stock.png')
    plt.close() # close current figure

# list of predefined functions, each described with its name, description, and parameters
# gpt now can know what to expect from its user by looking at these functions
functions = [
    {
        'name': 'get_stock_price',
        'description': 'gets the latest stock price given the ticker symbol of a company.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'the stock ticker symbol for a given company'
                }
            },
            'required': ['ticker']
        }
    },
    {
        'name': "calculate_sma",
        'description': 'calculates the simple moving average for a given stock ticker and a window.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'the stock ticker symbol for a given company'
                },
                'window': {
                    'type': 'integer',
                    'description': 'the timeframe to consider when calculating the simple moving average'
                }
            },
            'required': ['ticker', 'window'],
        },
    },
    {
        'name': "calculate_ema",
        'description': 'calculates the exponential moving average for a given stock ticker and a window.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'the stock ticker symbol for a given company'
                },
                'window': {
                    'type': 'integer',
                    'description': 'the timeframe to consider when calculating the exponential moving average'
                }
            },
            'required': ['ticker', 'window'],
        },
    },
    {
        'name': "calculate_rsi",
        'description': 'calculates the relative strength index for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'the stock ticker symbol for a given company.'
                },
            },
            'required': ['ticker'],
        },
    },
    {
        'name': "calculate_macd",
        'description': 'calculates the moving average convergence divergence for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'the stock ticker symbol for a given company.'
                },
            },
            'required': ['ticker'],
        },
    },
    {
        'name': "plot_stock_price",
        'description': 'plot the stock price for the last year given the ticker symbol of a company.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'the stock ticker symbol for a given company.'
                },
            },
            'required': ['ticker'],
        },
    },
]

available_functions = { # allows easy access to these functions to gpt without immediately invoking
    'get_stock_price': get_stock_price,  # Passes function reference for 'get_stock_price'
    'calculate_sma': calculate_sma,
    'calculate_ema': calculate_ema,
    'calculate_rsi': calculate_rsi,
    'calculate_macd': calculate_macd,
    'plot_stock_price': plot_stock_price
}

if 'messages' not in st.session_state:
    st.session_state['messages'] = [] # if not in session state creates key assigns empty list

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': f'{user_input}'}) # appends user input to messages in session state with role as user
        response = openai.ChatCompletion.create( # creates chat based response
            model='gpt-3.5-turbo-0613',
            messages=st.session_state['messages'],
            functions=functions,
            function_call='auto'
        )

        response_message = response['choices'][0]['message'] # retrieves response message

        # Check if the response message contains a function call.
        if response_message.get('function_call'):
            function_name = response_message['function_call']['name'] # extract function name and arguments from the function call
            function_args = json.loads(response_message['function_call']['arguments'])

            if function_name in ['get_stock_price', 'calculate_rsi', 'calculate_macd', 'plot_stock_price']: # prepares arguments dictionary based on function type.
                args_dict = {'ticker': function_args.get('ticker')}
            else:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}

            function_to_call = available_functions[function_name] #  fetchs reference to function to be called
            function_response = function_to_call(**args_dict) # calls function and gets the response

            if function_name == 'plot_stock_price':
                st.image('stock.png')
            else:
                st.session_state['messages'].append(response_message) # append response message and function details to messages
                st.session_state['messages'].append(
                    {
                        'role': 'function',
                        'name': function_name,
                        'content': function_response
                    }
                )

                second_response = openai.ChatCompletion.create( # generates second response after using gpt
                    model='gpt-3.5-turbo-0613',
                    messages=st.session_state['messages']
                )
                st.text(second_response['choices'][0]['message']['content']) # displays gpts text response
                st.session_state['messages'].append(
                    {'role': 'assistant', 'content': second_response['choices'][0]['message']['content']})
        else:
            st.text(response_message['content']) # displays response if response doesnt involve function call

            st.session_state['messages'].append({'role': 'assistant', 'content': response_message['content']})
    except Exception as e:
        raise e

