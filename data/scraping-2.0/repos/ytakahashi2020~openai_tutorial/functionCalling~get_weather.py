from openai import OpenAI
import json

client = OpenAI()

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"}]
    tools = [
        {
            "type": "function",
            "function": {
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
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )

    # print("tool_calls",
    #       response.choices[0].message.tool_calls)
    # print("\n")

    print("ID: ",
          response.choices[0].message.tool_calls[0].id)
    print("\n")
    print("タイプ: ",
          response.choices[0].message.tool_calls[0].type)
    print("\n")
    print("関数名: ",
          response.choices[0].message.tool_calls[0].function.name)
    print("\n")
    print("引数: ",
          response.choices[0].message.tool_calls[0].function.arguments)
    print("\n")

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        # extend conversation with assistant's reply
        print("元メッセージ: ", messages)
        print("\n")
        messages.append(response_message)
        print("メッセージ（関数提案の追加）: ", messages)
        print("\n")
        print("追加部分: ", response_message)
        print("\n")
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

            # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        print("メッセージ（関数結果の追加）: ", messages)
        print("\n")

        print("追加部分: ", {
            "tool_call_id": tool_call.id,
            "role": "tool",
                    "name": function_name,
                    "content": function_response,
        })
        print("\n")
        return second_response

    # print("messages", messages)


# print(run_conversation())


print(run_conversation().choices[0].message.content)
