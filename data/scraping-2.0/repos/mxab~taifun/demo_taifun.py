import os
import openai
import random
from taifun import Taifun


openai.api_key_path = os.path.expanduser("~") + "/.openai_api_key"

taifun = Taifun()


@taifun.fn()
def weather_forcast(location: str) -> str:
    """
    Get the weather forcast for a given location

    Parameters
    ----------
    location: str
        the location to get the weather forcast for

    """

    # random weather
    weather = random.choice(["sunny", "rainy", "cloudy", "snowy"])

    return f"The weather in {location} is {weather}"


messages = [
    {
        "role": "user",
        "content": "Is it rainingy in berlin today?",
    },
]

# export functions as json schema dict for openai
functions = taifun.functions_as_dict()


result = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    functions=functions,
    function_call="auto",
)
response_message = result["choices"][0]["message"]

print(f"assistant: {response_message['content']}")

function_call = response_message.get("function_call")

messages.append(response_message)
if function_call is not None:
    # handle the function call
    function_response = taifun.handle_function_call(function_call)

    # responed with the function response
    print(f"function response: {function_response}")
    messages.append(
        {
            "role": "function",
            "name": function_call["name"],
            "content": function_response,
        }
    )

    result2 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_message2 = result2["choices"][0]["message"]
    print(f"assistant: {response_message2['content']}")
