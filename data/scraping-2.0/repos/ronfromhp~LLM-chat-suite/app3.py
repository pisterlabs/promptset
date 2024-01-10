from openai import AsyncClient
import json
import ast
import os
import chainlit as cl
from chainlit.prompt import Prompt, PromptMessage

openai_client = AsyncClient(api_key=os.environ.get("OPENAI_API_KEY"))

MAX_ITER = 5


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    unit = unit or "Farenheit"
    weather_info = {
        "location": location,
        "temperature": "60",
        "unit": unit,
        "forecast": ["windy"],
    }
    
    print(weather_info)

    return json.dumps(weather_info)

def get_taxi_booking_information(
    pickup_location, dropoff_location, pickup_time, number_of_passengers
):
    """Get the taxi booking information"""
    booking_info = {
        "pickup_location": pickup_location,
        "dropoff_location": dropoff_location,
        "pickup_time": pickup_time,
        "number_of_passengers": number_of_passengers,
    }
    
    print(booking_info)

    return json.dumps(booking_info)

def get_user_information():
    """Get the user information, such as the name, city, and state, home address, work address, etc. This function can be called at the start of the conversation to get the user information."""
    user_info = {
        "name": "John Doe",
        "city": "San Francisco",
        "state": "CA",
        "home_address": "123 Main St, San Francisco, CA",
        "work_address": "456 Main St, San Francisco, CA",
    }

    return json.dumps(user_info)


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
    },
    {
        "name": "get_taxi_booking_information",
        "description": "Get the taxi booking information. Ask the questions one by one as the user is a senior citizen and may not be able to answer all the questions at once",
        "parameters": {
            "type": "object",
            "properties": {
                "pickup_location": {
                    "type": "string",
                    "description": "The pickup location in the city",
                },
                "dropoff_location": {
                    "type": "string",
                    "description": "The dropoff location. Should always be a valid address.",
                },
                "pickup_time": {
                    "type": "string",
                    "description": "The pickup time. should always be a valid time. if not specified then the current time is given as NOW",
                },
                "number_of_passengers": {
                    "type": "integer",
                    "description": "The number of passengers",
                }
            },
            "required": ["pickup_location", "dropoff_location", "pickup_time", "number_of_passengers"],
        },
    },
    {
        "name": "get_user_information",
        "description": "Get the user information, such as the name, city, and state, home address, work address, etc. This function can be called at the start of the conversation to get the user information.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }
]


async def process_new_delta(
    new_delta, openai_message, content_ui_message, function_ui_message
):
    if new_delta.role:
        openai_message["role"] = new_delta.role

    new_content = new_delta.content or ""
    openai_message["content"] += new_content
    await content_ui_message.stream_token(new_content)
    if new_delta.function_call:
        if new_delta.function_call.name:
            openai_message["function_call"] = {"name": new_delta.function_call.name}
            await content_ui_message.send()
            function_ui_message = cl.Message(
                author=new_delta.function_call.name,
                content="",
                parent_id=content_ui_message.id,
                language="json",
            )
            await function_ui_message.stream_token(new_delta.function_call.name)

        if new_delta.function_call.arguments:
            if "arguments" not in openai_message["function_call"]:
                openai_message["function_call"]["arguments"] = ""
            openai_message["function_call"][
                "arguments"
            ] += new_delta.function_call.arguments
            await function_ui_message.stream_token(new_delta.function_call.arguments)
    return openai_message, content_ui_message, function_ui_message


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful coding assistant. You are specialised to work with coders and provide detailed implementations"}],
    )


@cl.on_message
async def run_conversation(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        # OpenAI call
        openai_message = {"role": "", "content": ""}
        function_ui_message = None
        content_ui_message = cl.Message(content="")

        await content_ui_message.send()

        settings = {
            "model": "gpt-4-1106-preview",
            "function_call": "auto",
            "functions": functions,
            "temperature": 0,
        }

        stream = await openai_client.chat.completions.create(
            messages=message_history, stream=True, **settings
        )

        finish_reason = None

        async for part in stream:
            new_delta = part.choices[0].delta
            (
                openai_message,
                content_ui_message,
                function_ui_message,
            ) = await process_new_delta(
                new_delta, openai_message, content_ui_message, function_ui_message
            )
            finish_reason = part.choices[0].finish_reason

        prompt = Prompt(
            provider="openai-chat",
            messages=[
                PromptMessage(
                    formatted=m["content"], name=m.get("name"), role=m["role"]
                )
                for m in message_history
            ],
            settings=settings,
            completion=content_ui_message.content,
        )
        content_ui_message.prompt = prompt
        await content_ui_message.update()

        message_history.append(openai_message)
        if function_ui_message is not None:
            await function_ui_message.send()

        if finish_reason == "stop":
            break

        elif finish_reason != "function_call":
            raise ValueError(finish_reason)

        # if code arrives here, it means there is a function call
        function_name = openai_message.get("function_call", {}).get("name")
        arguments = ast.literal_eval(
            openai_message.get("function_call", {}).get("arguments")
        )

        if function_name == "get_current_weather":
            function_response = get_current_weather(
                location=arguments.get("location"),
                unit=arguments.get("unit"),
            )
        elif function_name == "get_taxi_booking_information":
            function_response = get_taxi_booking_information(
                pickup_location=arguments.get("pickup_location"),
                dropoff_location=arguments.get("dropoff_location"),
                pickup_time=arguments.get("pickup_time"),
                number_of_passengers=arguments.get("number_of_passengers"),
            )
        elif function_name == "get_user_information":
            function_response = get_user_information()
        

        message_history.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )

        await cl.Message(
            author=function_name,
            content=str(function_response),
            language="json",
            parent_id=content_ui_message.id,
        ).send()

        cur_iter += 1