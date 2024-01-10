import json
import os
import sys
import openai

openai.api_type = "azure"
openai.api_base = "https://test-iit-hsp-instance-20230531.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("AZURE_API_KEY")
azureEngine = "gpt35-turbo16k-v0613"

# Example for robot navigation
# In production, this could be your backend API or an external API
def go_to(location):
    """Moves the robot to a target location"""
    # Write here the code for the actual navigation
    ###############################################

    nav_info = {
        "location": location,
        "result": "target reached"
    }
    return json.dumps(nav_info)

# Example for finding objects
# In production, this could be your backend API or an external API
def find(object, location=None):
    """Looks for an object (in a certain location, if specified)"""
    nav_info = "No navigation needed"
    if location is not None:
        # Write here the code for the actual navigation
        ###############################################
        nav_info = {
            "location": location,
            "result": "target reached"
        }
    # Write here the code needed to look for an object
    ###################################################

    search_info = {
        "object": object,
        "result": "{} found".format(object),
        "object_pos": {
            "x": 1.0,
            "y": -0.5,
            "z": 0.8,
        },
        "navigation": nav_info
    }

    return json.dumps(search_info)

def run_conversation(engineIn,request):
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": request}]
    functions = [
        {
            "name": "go_to",
            "description": "Goes to a certain location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to move the robot to",
                    }
                },
                "required": ["location"],
            },
        }, {
            "name": "find",
            "description": "Looks for an object (in a certain location, if specified)",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "The object to look for",
                    },
                    "location": {
                        "type": "string",
                        "description": "The location to move the robot to",
                    }
                },
                "required": ["object"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        engine=engineIn,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    print("\n\n{}\n\n".format(response))

    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "go_to": go_to,
            "find": find,
        }
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            **function_args,
        )

        # Step 4: send the info on the function call and function response to GPT
        #messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        second_response = openai.ChatCompletion.create(
            engine=engineIn,
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        #return second_response
        return second_response["choices"][0]["message"]["content"]
    else:
        return response["choices"][0]["message"]["content"]

if len(sys.argv) == 2:
    print(run_conversation(azureEngine,sys.argv[1]))
else:
    print("The actual request is missing")
