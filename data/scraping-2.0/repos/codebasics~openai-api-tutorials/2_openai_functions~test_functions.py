import openai
from secret_key import openai_api_key
import json

openai.api_key = openai_api_key

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


response_message = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "What is the weather like in boston?"}
                ],
                functions=[
                    {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["location"]
                        }
                    }
                ]
            )

if response_message.get('function_call'):
    function_name = response_message['function_call']["name"]
    function_args = json.loads(response_message['function_call']["arguments"])

    available_functions = {
        'get_current_weather': get_current_weather
    }

    function_to_call = available_functions[function_name]

    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit")
    )

