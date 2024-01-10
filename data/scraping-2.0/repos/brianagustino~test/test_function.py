import openai
import json

def get_current_weather(location, units="fahrenheit"):
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": units,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

def run_conversaations():
    messages = [{"role":"user", "content":"What's the weather like in Boston?"}]
    functions = [
        {
            "name":"get_current_weather",
            "description":"get weather of chosen location",
            "parameters": {
                "type":"object",
                "properties": {
                    "location": {
                        "type":"string",
                        "description":"City and State",
                    },
                    "unit": {
                        "type":"string",
                        "enum": ["celcius", "fahrenheit"],
                    },
                },
                "required":["location"]
            },
        }
    ]
    openai.api_key = "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"

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0613",
        messages = messages,
        functions = functions,
        function_call = "auto",
        max_tokens = 50,
    )
    response_message = response["choices"][0]["message"]
    print(response)
    #print(response_message)
    
    if response_message.get("function_call"):
        #Make a Dictionary of the available Functions
        available_functions = {
            "get_current_weather":get_current_weather,
        }
        #Get the Function name from the response of AI
        function_name = response_message["function_call"]["name"]
        #Get the function to call from the Dictionary of functions
        function_to_call = available_functions[function_name]
        #Get the arguments from the response of AI
        function_args = json.loads(response_message["function_call"]["arguments"])
        #Call the directed function from AI response
        function_response = function_to_call(
            location = function_args.get("location"),
            units = function_args.get("units"),
        )
        #print("This is function response")
        #print(function_response)
        #Append response of AI into messages (up-to-date) of assistant's reply
        messages.append(response_message)
        #Add message of triggered function
        messages.append(
            {
                "role":"function",
                "name":function_name,
                "content":function_response,
            }
        )
        #Update the function response to the AI
        second_response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0613",
            messages = messages,
        )
        return second_response


print(run_conversaations())