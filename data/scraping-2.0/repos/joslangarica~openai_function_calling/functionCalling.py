import os
import openai
from clima import get_clima
import json


openai.api_key = os.getenv('OPENAI_API_KEY')

weather_api_key = os.getenv("WEATHER_API_KEY")


def run_conversation():
    messages = [
        {"role": "user", "content": "What's the weather like in Mexico City?"}]
    # describe function
    function_definition = [
        {
            "name": "get_clima",
            "description": "Get the current weather in a given location",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                "required": ["location"],
            },
        }
    ]

    first_call = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=messages,
        functions=function_definition,
        function_call='auto'
    )

    resultado = first_call.choices[0].message

    params = json.loads(resultado.function_call.arguments)
    location = json.loads(resultado.function_call.arguments).get('location')
    function_to_call = resultado.function_call.name
    print("First call RESULTADO: ", resultado)
    print("PARAMS: ", params),
    print("LOCATION: ", location)
    print('FUNCTION THAT AI DECIDED TO CALL: ', function_to_call)

    if resultado.get('function_call'):
        available_functions = {
            'get_clima': get_clima
        }
        function_name = resultado['function_call']['name']
        function_to_call = available_functions[function_name]
        function_args = params,
        function_response = function_to_call(
            api_key=os.getenv('WEATHER_API_KEY'),
            location=location
        )

        messages.append(resultado)
        messages.append(
            {
                'role': 'function',
                'name': function_name,
                'content': function_response,
            }
        )

        second_response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613',
            messages=messages,
        )

        print("WEATHER API RESULT FOR TESTING: ",
              get_clima(weather_api_key, location))
        return second_response


print(run_conversation())
