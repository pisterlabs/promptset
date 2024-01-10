import openai

messages = [
    {
        "role": "user",
        "content": input("Message: ")
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "play_song",
            "description": "Play a song",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "The name of the song to play",
                    },
                },
                "required": ["song_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "light_dimmer",
            "description": "Adjust the light dimmer from 0-100",
            "parameters": {
                "type": "object",
                "properties": {
                    "brightness": {
                        "type": "number",
                        "description": "The brightness from 0-100",
                    },
                },
                "required": ["brightness"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "order_food",
            "description": "Order food from a restaurant",
            "parameters": {
                "type": "object",
                "properties": {
                    "dish_name": {
                        "type": "string",
                        "description": "The name of the dish to order",
                    },
                    "count": {
                        "type": "number",
                        "description": "The number of dishes to order",
                    },
                },
                "required": ["dish_name", "count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send a text message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "The name of the contact",
                    },
                    "message": {
                        "type": "string",
                        "description": "The text message content",
                    },
                },
                "required": ["contact_name", "message"],
            },
        },
    }
]

response = openai.chat.completions.create(
    #model="gpt-3.5-turbo-1106",
    model="gpt-4-1106-preview",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

messages.append(response_message)

for tool_call in tool_calls:
    function_name = tool_call.function.name
    arguments = tool_call.function.arguments
    print(f"Called '{function_name}' with args {arguments}")
