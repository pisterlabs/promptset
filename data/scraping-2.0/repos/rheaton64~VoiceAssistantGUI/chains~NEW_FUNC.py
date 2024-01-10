instructions="""# Function calling

In an API call, you can describe functions to gpt-3.5-turbo-0613 and gpt-4-0613, and have the model intelligently choose to output a JSON object containing arguments to call those functions. The Chat Completions API does not call the function; instead, the model generates JSON that you can use to call the function in your code.

The latest models (gpt-3.5-turbo-0613 and gpt-4-0613) have been fine-tuned to both detect when a function should to be called (depending on the input) and to respond with JSON that adheres to the function signature. With this capability also comes potential risks. We strongly recommend building in user confirmation flows before taking actions that impact the world on behalf of users (sending an email, posting something online, making a purchase, etc).

Under the hood, functions are injected into the system message in a syntax the model has been trained on. This means functions count against the model's context limit and are billed as input tokens. If running into context limits, we suggest limiting the number of functions or the length of documentation you provide for function parameters.
Function calling allows you to more reliably get structured data back from the model. For example, you can:

Create chatbots that answer questions by calling external APIs (e.g. like ChatGPT Plugins)
e.g. define functions like send_email(to: string, body: string), or get_current_weather(location: string, unit: 'celsius' | 'fahrenheit')
Convert natural language into API calls
e.g. convert "Who are my top customers?" to get_customers(min_revenue: int, created_before: string, limit: int) and call your internal API
Extract structured data from text
e.g. define a function called extract_data(name: string, birthday: string), or sql_query(query: string)
...and much more!

The basic sequence of steps for function calling is as follows:

Call the model with the user query and a set of functions defined in the functions parameter.
The model can choose to call a function; if so, the content will be a stringified JSON object adhering to your custom schema (note: the model may generate invalid JSON or hallucinate parameters).
Parse the string into JSON in your code, and call your function with the provided arguments if they exist.
Call the model again by appending the function response as a new message, and let the model summarize the results back to the user.

You will not be calling a function, but instead creating a new function for another model to call
The steps and instructions for creating a new function are as follows.

# Defining custom functions example

1. Define custom functions

```
import yfinance as yf
from datetime import datetime, timedelta


def get_current_stock_price(ticker):
    \"\"\"Method to get current stock price\"\"\"

    ticker_data = yf.Ticker(ticker)
    recent = ticker_data.history(period="1d")
    return {"price": recent.iloc[0]["Close"], "currency": ticker_data.info["currency"]}


def get_stock_performance(ticker, days):
    \"\"\"Method to get stock price change in percentage\"\"\"

    past_date = datetime.today() - timedelta(days=days)
    ticker_data = yf.Ticker(ticker)
    history = ticker_data.history(start=past_date)
    old_price = history.iloc[0]["Close"]
    current_price = history.iloc[-1]["Close"]
    return {"percent_change": ((current_price - old_price) / old_price) * 100}
```

2. Make custom tools using the BaseFunction class
```
from typing import Type
from pydantic import BaseModel, Field
from functions.BaseFunction import BaseFunction
from langchain.tools import BaseTool

# The following is the definition of the BaseFunction for reference only, you don't have to rewrite it every time

class BaseFunction:
    def __init__(self, name):
        self.name = name
        self.args_schema = BaseModel  # Use Pydantic's BaseModel as a placeholder

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")

# end BaseFunction definition


class CurrentStockPriceTool(BaseFunction):
    class CurrentStockPriceInput(BaseModel): # This is the args schema of the function, with a desc for each field for the model to read
        ticker: str = Field(description="Ticker symbol of the stock")

    def __init__(self):
        super().__init__("get_current_stock_price")
        self.args_schema = self.CurrentStockPriceInput

    def run(self, ticker: str): # This should call the previously defined function
        price_response = get_current_stock_price(ticker)
        return price_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_current_stock_price does not support async") # Try to implement this if possible

# Repeat the same process for the StockPerformanceTool class

class StockPerformanceTool(BaseFunction):
    class StockPercentChangeInput(BaseModel):
        ticker: str = Field(description="Ticker symbol of the stock")
        days: int = Field(description="Timedelta days to get past date from current date")

    def __init__(self):
        super().__init__("get_stock_performance")
        self.args_schema = self.StockPercentChangeInput

    def run(self, ticker: str, days: int):
        response = get_stock_performance(ticker, days)
        return response

    def _arun(self, ticker: str, days: int):
        raise NotImplementedError("get_stock_performance does not support async")
```"""