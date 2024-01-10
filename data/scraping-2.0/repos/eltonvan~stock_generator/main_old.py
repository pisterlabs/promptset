import streamlit as st
import json
from openai import OpenAI

client = OpenAI(api_key=open("API_KEY", "r").read())
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf


def get_stock_price(ticker):
    """Gets the current stock price of a stock ticker"""
    return str(yf.Ticker(ticker).history(period="1y").iloc[-1].Close)


def calculate_SMA(ticker, window):
    """Calculates the Simple Moving Average of a stock ticker"""
    data = yf.Ticker(ticker).history(period="1y").close
    return str(data.rolling(window=window).mean().iloc[-1])


def calculate_EMA(ticker, window):
    """Calculates the Exponential Moving Average of a stock ticker"""
    data = yf.Ticker(ticker).history(period="1y").close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])


def calculate_RSI(ticker):
    """Calculates the Relative Strength Index of a stock ticker"""
    data = yf.Ticker(ticker).history(period="1y").close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])


def calculate_MACD(ticker):
    """Calculates the Moving Average Convergence Divergence of a stock ticker"""
    data = yf.Ticker(ticker).history(period="1y").close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()

    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f"{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}"


def plot_stock_price(ticker):
    """Plots the stock price of a stock ticker"""
    data = yf.Ticker(ticker).history(period="1y")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title("{ticker} Stock Price over last year")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.grid(True)
    plt.savefig("stock.png")
    plt.close()


functions = [
    {
        "name": "Get Stock Price",
        "description": "Gets the latest stock price given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "Calculate_SMA",
        "description": "Calculates the Simple Moving Average of a stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                },
                "window": {
                    "type": "integer",
                    "description": "The window size for the Simple Moving Average.",
                },
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "Calculate_EMA",
        "description": "Calculates the Exponential Moving Average of a stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                },
                "window": {
                    "type": "integer",
                    "description": "The time frame to consider when calculating the Exponential Moving Average.",
                },
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "Calculate_RSI",
        "description": "Calculates the Relative Strength Index of a stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "Calculate_MACD",
        "description": "Calculates the Moving Average Convergence Divergence of a stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "Plot_Stock_Price",
        "description": "Plots the stock price of a stock ticker for the last year.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (for example AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
]

available_functions = {
    "Get Stock Price": get_stock_price,
    "Calculate_SMA": calculate_SMA,
    "Calculate_EMA": calculate_EMA,
    "Calculate_RSI": calculate_RSI,
    "Calculate_MACD": calculate_MACD,
    "Plot_Stock_Price": plot_stock_price,
}


if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Stock Market API assistant")

user_input = st.text_input("Enter your query here:")
if user_input:
    try:
        st.session_state["messages"].append(
            {"role": "user", "content": f"{user_input}"}
        )

        prompt = {
            "engine": "gpt-3.5-turbo-0613",
            "prompt": st.session_state["messages"],
            "temperature": 0.7,
            "max_tokens": 150,
            "stream": False,
            "functions": functions,
            "function_call": "auto",
        }

        response = client.completions.create(**prompt)

        response_message = response["choices"][0]["message"]
        print(response)

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args = json.loads(response_message["function_call"]["arguments"])
            if function_name in [
                "get_stock_price",
                "calculate_MACD",
                "plot_stock_price",
                "calculate_RSI",
            ]:
                args_dict = {"ticker": function_args.get("ticker")}
            elif function_name in ["calculate_SMA", "calculate_EMA"]:
                args_dict = {
                    "ticker": function_args.get("ticker"),
                    "window": function_args.get("window"),
                }

            functions_to_call = available_functions[function_name]
            function_response = functions_to_call(**args_dict)

            if function_name == "plot_stock_price":
                st.image("stock.png")

            else:
                st.session_state["messages"].append(response_message)
                st.session_state["messages"].append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )

                second_response = client.completions.create(
                    engine="gpt-3.5-turbo-0613",
                    prompt=st.session_state["messages"],
                    temperature=0.7,
                    max_tokens=150,
                    stream=False,
                )
                st.text(second_response["choices"][0]["message"]["content"])
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": second_response["choices"][0]["message"]["content"],
                    }
                )

        else:
            st.text(response_message["content"])
            st.session_state["messages"].append(
                {"role": "assistant", "content": response_message["content"]}
            )

    except Exception as e:
        print(e)
