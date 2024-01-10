import json
import openai
from decouple import config

from apis.weather import get_current_weather, get_weather_forecast
from func_descriptions.weather import (
    describe_get_current_weather,
    describe_get_weather_forecast,
)
from utils.printer import ColorPrinter as Printer

openai.api_key = config("CHATGPT_API_KEY")


def ask_chat_gpt(query):
    messages = [
        {"role": "user", "content": query},
    ]

    functions = [describe_get_current_weather, describe_get_weather_forecast]

    first_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default
    )["choices"][0]["message"]
    messages.append(first_response)

    if first_response.get("function_call"):
        available_functions = {
            "get_current_weather_in_location": get_current_weather,
            "get_weather_forecast_in_location": get_weather_forecast,
        }
        function_name = first_response["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(first_response["function_call"]["arguments"])
        function_response = function_to_call(**function_args)

        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )["choices"][0]["message"]
        messages.append(second_response)

        Printer.color_print(messages)
        return second_response["content"]

    Printer.color_print(messages)
    return first_response["content"]


print(ask_chat_gpt("What is the weather forecast in Leipzig for the coming 3 days?"))
