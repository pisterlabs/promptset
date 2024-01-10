import openai
import json
openai.api_key = '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'


#Testing openAI's access within the program
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

def run_conversation():
    messages = [{"role": "user", "content": "Provide personalized recommendation for international students staying off campus near Georgia State University. The response must be in specific bullet points starting from the time a student arrives on the campus to finding suitable accommodations, understanding local bills and contract terms, setting up essential banking services, and accessing additional resources."}]
    functions = [
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
        }
    ]
    response = openai.ChatCompletion.create(
        model="text-davinci-002",
        messages=messages,
        functions=functions,
        function_call="auto",  
    )
    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "get_current_weather": get_current_weather,
        }  
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(response_message)  
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  
        second_response = openai.ChatCompletion.create(
            model="text-davinci-002",
            messages=messages,
        )  
        return second_response

print(run_conversation())