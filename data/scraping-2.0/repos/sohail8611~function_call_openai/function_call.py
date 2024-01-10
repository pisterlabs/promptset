import openai
import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

def sum_two_integers(num1,num2):
    """Get the current weather in a given location"""
    info = {
        "sum": int(num1) + int(num2),
        
    }
    return json.dumps(info)


# Step 1, send model the user query and what functions it has access to
def run_conversation():
    input_query = "hey how are you doing?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": input_query}],
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
                    "required": ["location"],
                },
            },

            {
                "name": "sum_two_integers",
                "description": "Return the sum of two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num1": {
                            "type": "string",
                            "description": "any positive integer",
                        },
                        "num2": {
                            "type": "string",
                            "description": "any positive integer",
                        },
                    },
                    "required": ["num1","num2"],
                },
            }
        ],
        function_call="auto",
    )
    print(response["choices"][0]["message"])
    message = response["choices"][0]["message"]

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        function_name = message["function_call"]["name"]

        if function_name == 'get_current_weather':

            # Step 3, call the function
            # Note: the JSON response from the model may not be valid JSON
            function_response = get_current_weather(
                location=json.loads(message['function_call']['arguments'])['location'],
            )
        else:
            function_response = sum_two_integers(
                num1=json.loads(message['function_call']['arguments'])['num1'],
                num2=json.loads(message['function_call']['arguments'])['num2'],
            )


        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": input_query},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response
    else:
        print(response)

print(run_conversation()['choices'][0]['message']['content'])