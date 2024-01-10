import streamlit as st
import openai
import time
import json
import sys

import pytz
from datetime import datetime

import pandas as pd
import json

import math

import inspect


st.title("AI Assistant - Stock data Analysis - Function Calling")


def get_current_time(location):
    try:
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except:
        return "Sorry, I couldn't find the timezone for that location."


def get_stock_market_data(index):
    available_indices = [
        "S&P 500",
        "NASDAQ Composite",
        "Dow Jones Industrial Average",
        "Financial Times Stock Exchange 100 Index",
    ]

    if index not in available_indices:
        return "Invalid index. Please choose from 'S&P 500', 'NASDAQ Composite', 'Dow Jones Industrial Average', 'Financial Times Stock Exchange 100 Index'."

    # Read the CSV file
    data = pd.read_csv("stock_data.csv")

    # Filter data for the given index
    data_filtered = data[data["Index"] == index]

    # Remove 'Index' column
    data_filtered = data_filtered.drop(columns=["Index"])

    # Convert the DataFrame into a dictionary
    hist_dict = data_filtered.to_dict()

    for key, value_dict in hist_dict.items():
        hist_dict[key] = {k: v for k, v in value_dict.items()}

    return json.dumps(hist_dict)


def calculator(num1, num2, operator):
    if operator == "+":
        return str(num1 + num2)
    elif operator == "-":
        return str(num1 - num2)
    elif operator == "*":
        return str(num1 * num2)
    elif operator == "/":
        return str(num1 / num2)
    elif operator == "**":
        return str(num1**num2)
    elif operator == "sqrt":
        return str(math.sqrt(num1))
    else:
        return "Invalid operator"


functions = [
    {
        "name": "get_current_time",
        "description": "Get the current time in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location name. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_stock_market_data",
        "description": "Get the stock market data for a given index",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "enum": [
                        "S&P 500",
                        "NASDAQ Composite",
                        "Dow Jones Industrial Average",
                        "Financial Times Stock Exchange 100 Index",
                    ],
                },
            },
            "required": ["index"],
        },
    },
    {
        "name": "calculator",
        "description": "A simple calculator used to perform basic arithmetic operations",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "number"},
                "num2": {"type": "number"},
                "operator": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/", "**", "sqrt"],
                },
            },
            "required": ["num1", "num2", "operator"],
        },
    },
]

available_functions = {
    "get_current_time": get_current_time,
    "get_stock_market_data": get_stock_market_data,
    "calculator": calculator,
}


# helper method used to check if the correct arguments are provided to a function
def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True


def init_config():
    if "deployment_name" not in st.session_state:
        sys.path.insert(0, "../../0_common_config")
        from config_data import get_deployment_name_turbo, set_environment_details_turbo

        st.session_state["deployment_name"] = get_deployment_name_turbo()
        set_environment_details_turbo()
        print(
            "deployment_name",
            st.session_state["deployment_name"],
            "\nopenai.api_base",
            openai.api_base,
            "\nopenai.api_type",
            openai.api_type,
        )
    return st.session_state["deployment_name"]


if "messages" not in st.session_state:
    st.session_state.messages = []
    system_prompt = ""
    with open("metaprompt-1.txt", "r") as file:
        # system_prompt = file.read().replace('\n', '')
        system_prompt = file.read()
        st.session_state.messages.append({"role": "system", "content": system_prompt})
        st.text_area(label="System Prompt", value=system_prompt, height=500)

counter = 0
full_response = ""

for message in st.session_state.messages:
    if counter == 0:
        counter += 1
        continue
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Hello! What would you like me to help you with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    asst_response = ""
    messages = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = openai.ChatCompletion.create(
            engine=init_config(),
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        counter=0
        # Step 2: check if GPT wanted to call a function
        while response["choices"][0]["finish_reason"] == "function_call":
            response_message = response["choices"][0]["message"]
            print("Recommended Function call:")
            print(response_message.get("function_call"))
            print()
            counter += 1


            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors

            function_name = response_message["function_call"]["name"]

            # verify function exists
            if function_name not in available_functions:
                message_placeholder.markdown(
                    "Function " + function_name + " does not exist" + "▌"
                )
                # return "Function " + function_name + " does not exist"
            function_to_call = available_functions[function_name]

            # verify function has correct number of arguments
            function_args = json.loads(response_message["function_call"]["arguments"])
            if check_args(function_to_call, function_args) is False:
                message_placeholder.markdown(
                    "Invalid number of arguments for function: " + function_name + "▌"
                )
                # return "Invalid number of arguments for function: " + function_name
            
            full_response += ':orange[Function Call #'+ str(counter) + "-->]  \n name: " + function_name + "  \n arguments: ```" + str(function_args) + "```  \n"
            
            function_response = function_to_call(**function_args)

            print("Output of function call:")
            print(function_response)
            print()
            full_response += ':red[Response from Function Call # '+ str(counter) + "-->]  \n```"  + str(function_response)+ "```  \n" +"......................  \n"
            message_placeholder.markdown(full_response)

            # Step 4: send the info on the function call and function response to GPT

            # adding assistant response to messages

            messages.append(
                {
                    "role": response_message["role"],
                    "name": response_message["function_call"]["name"],
                    "content": response_message["function_call"]["arguments"],
                }
            )

            # adding function response to messages
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response

            print("Messages in next request:")
            # for message in messages:
            #     print(message)
            # print()

            response = openai.ChatCompletion.create(
                messages=messages,
                engine=init_config(),
                function_call="auto",
                functions=functions,
                temperature=0,
            )  # get a new response from GPT where it can see the function response
            asst_response = response

        llm_response = asst_response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": llm_response})
        message_placeholder.markdown(full_response + "  \n" + ":blue[Final response to user question -->]  \n" + llm_response + "  \n")
        st.session_state.messages.clear()
