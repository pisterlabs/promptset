# The json module  can be used to parse JSON strings into Python objects and serialize Python objects into JSON strings.
import json 

# The logging module provides a flexible framework for emitting log messages from Python programs. 
# It allows you to record events and error messages at different levels of severity, such as DEBUG, INFO, WARNING, ERROR, and CRITICAL. 
import logging

# By importing the operator module, you can use its functions to perform operations such as arithmetic, item access, comparison, logical operations, and more. It provides a convenient way to work with operators as functions in Python.
import operator

# By importing the sys module, you can use its functions and variables to perform tasks such as accessing command-line arguments, interacting with the standard input/output/error streams, and getting information about the Python interpreter.
import sys

# By importing the datetime module, you can use its classes and functions to perform operations such as creating date and time objects, extracting specific components from dates and times, performing arithmetic operations on dates and times, and formatting dates and times into strings.
import datetime

# By importing the openai module, you can use its functions and classes to make requests to the OpenAI API, such as generating text, completing prompts, and performing other natural language processing tasks using the power of the GPT-3 and GPT4 models.
import openai

# The yfinance module is a popular Python library that provides a simple and convenient way to download historical market data from Yahoo Finance. 
# It allows you to fetch historical stock prices, dividend data, and other financial information for analysis and research purposes.
import yfinance as yf

# Assigning the current date to the variable TODAY in the format "YYYY/MM/DD".
TODAY = datetime.date.today().strftime("%Y/%m/%d")

# Configutation of the logging function. 
# The logging module provides a flexible framework for emitting log messages from Python programs.
# level=logging.WARNING sets the logging level to WARNING. This means that only log messages with a severity level of WARNING or higher (e.g., ERROR, CRITICAL)
# format="%(asctime)s %(message)s" sets the format of the log messages. In this case, it includes the timestamp of the log message (%(asctime)s) and the actual log message itself (%(message)s).
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

# Creating a logger object named logger using the getLogger() function from the logging module.
logger = logging.getLogger(__name__)

# Setting the logging level of the logger object to INFO.
# Log messages with severity levels such as INFO, WARNING, ERROR, and CRITICAL will be displayed.
logger.setLevel(logging.INFO)


# get_price function takes two parameters: symbol of type str and date of type str. It returns a float value.
# Logs an informational message using the logger object, indicating that the get_price function is being called and displaying the values of symbol and date.
# Using yf.download() function to download the historical stock data for the given symbol starting from the specified date. 
# Data fetched for a period of 1 day with a daily interval; progress is set to False.
# Returning the closing price to the "Close" column of history DataFrame; retrieving the value at the first row using iloc[0]. 
# # The item() method is used to convert the value to a float.

def get_price(symbol: str, date: str) -> float:
    logger.info(f"Calling get_price with {symbol=} and {date=}")

    history = yf.download(
        symbol, start=date, period="1d", interval="1d", progress=False
    )

    return history["Close"].iloc[0].item()

# calculate function takes three parameters: a and b of type float, and op of type str. It returns a float value.
# logging an informational message, indicating that the calculate function is being called; displaying the values of a, b, and op.
# It uses the getattr() function to dynamically retrieve the appropriate operator function from the operator module based on the value of op. 
# The getattr() function takes the operator module as the first argument and the value of op as the second argument.
# It calls the retrieved operator function with the arguments a and b and returns the result.

def calculate(a: float, b: float, op: str) -> float:
    logger.info(f"Calling calculate with {a=}, {b=} and {op=}")

    return getattr(operator, op)(a, b)


# Defining a dictionary named get_price_metadata that contains metadata information about a function called get_price.
# "name": "get_price" specifies the name of the function as "get_price".
# "description": "Get closing price of a financial instrument on a given date" provides a description of what the function does.
# "parameters" is an object that contains information about the function's parameters.
# "type": "object" indicates that the parameters are of type object.
# "properties" is an object that defines the properties of the object parameter.
# "symbol" is a property of type string that represents the ticker symbol of a financial instrument.
# "date" is a property of type string that represents the date in the format YYYY-MM-DD.
# "required": ["symbol", "date"] specifies that both the "symbol" and "date" properties are required when calling the get_price function.

get_price_metadata = {
    "name": "get_price",
    "description": "Get closing price of a financial instrument on a given date",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Ticker symbol of a financial instrument",
            },
            "date": {
                "type": "string",
                "description": "Date in the format YYYY-MM-DD",
            },
        },
        "required": ["symbol", "date"],
    },
}

# Defining a dictionary named calculate_metadata that contains metadata information about a function called calculate.
# "name": "calculate" specifies the name of the function as "calculate".
# "description": "General purpose calculator" provides a description of what the function does.
# "parameters" is an object that contains information about the function's parameters.
# "type": "object" indicates that the parameters are of type object.
# "properties" is an object that defines the properties of the object parameter.
# "a" is a property of type number that represents the first entry.
# "b" is a property of type number that represents the second entry.
# "op" is a property of type string that represents the binary operation. It is restricted to the values "mul", "add", "truediv", and "sub" using the enum keyword.
# "required": ["a", "b", "op"] specifies that all three properties, "a", "b", and "op", are required when calling the calculate function.

calculate_metadata = {
    "name": "calculate",
    "description": "General purpose calculator",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "First entry",
            },
            "b": {
                "type": "number",
                "description": "Second entry",
            },
            "op": {
                "type": "string",
                "enum": ["mul", "add", "truediv", "sub"],
                "description": "Binary operation",
            },
        },
        "required": ["a", "b", "op"],
    },
}

# Initializing a list called messages that contains dictionaries representing different messages.
# "content": sys.argv[1] assigns the value of the command-line argument at index 1 to the "content" key. This suggests that the user message is being passed as a command-line argument.

messages = [
    {"role": "user", "content": sys.argv[1]},
    {
        "role": "system",
        "content": "You are a helpful financial investor who overlooks the "
        f"performance of stocks. Today is {TODAY}. Note that the "
        "format of the date is YYYY/MM/DD",
    },
]

# while True in order to create an infinite loop; the loop continuously interacts with the OpenAI Chat API.
# includes the metadata information of the get_price and calculate functions, which might be used by the OpenAI model
# The response from the OpenAI model is stored in the response variable.
# The code then extracts the generated message from the response using response["choices"][0]["message"] and assigns it to the message variable.
# Finally, the message is appended to the messages list.

while True:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        messages=messages,
        functions=[get_price_metadata, calculate_metadata],
    )
    message = response["choices"][0]["message"]
    messages.append(message)

# Generating chat-based responses and continue generating responses until a generated message does not contain the substring "function_call".
# Checking if the substring "function_call" is not present in the variable message. 
# If the condition is true, it means that the generated message does not contain the substring "function_call".
# If the condition is true, the break statement is executed, which terminates the loop and exits the loop block.
    if "function_call" not in message:
        break

    # call custom functions
    # Extracting function_name and kwargs (keyword arguments) from the message object. 
    # The function_name is obtained from message["function_call"]["name"], 
    # The kwargs are obtained by parsing the JSON string stored in message["function_call"]["arguments"] using json.loads().
    function_name = message["function_call"]["name"]
    kwargs = json.loads(message["function_call"]["arguments"])

    # Checking if the value of function_name is "get_price".
    # If the condition is true, the get_price function is called with the keyword arguments stored in kwargs.
    # The output of the get_price function is converted to a string using str() and assigned to the output variable.
    # If the condition is false, the calculate function is called with the keyword arguments stored in kwargs.
    # The output of the calculate function is converted to a string using str() and assigned to the output variable.
    if function_name == "get_price":
        output = str(get_price(**kwargs))
    elif function_name == "calculate":
        output = str(calculate(**kwargs))
    else:
        raise ValueError
    
    # Finally, the output is appended to the messages list.
    messages.append({"role": "function", "name": function_name, "content": output})

# Creating a list comprehension [m["role"] for m in messages] that iterates over the messages list and extracts the value of the "role" key from each dictionary in the list. 
# The resulting list contains the values of the "role" key for each message.
# Accessing the last element of the messages list, and then retrieves the value associated with the "content" key from that dictionary. 
print("*" * 80)
print([m["role"] for m in messages])
print("*" * 80)
print(messages[-1]["content"])