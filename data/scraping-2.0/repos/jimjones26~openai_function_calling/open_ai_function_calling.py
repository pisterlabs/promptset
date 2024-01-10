# ------------------------------------------------------------------
# Import modules
# ------------------------------------------------------------------

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage

# ------------------------------------------------------------------
# Load opeaai api key from .env file
# ------------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------------
# Ask ChatGPT a question
# ------------------------------------------------------------------

completetion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "What is the next flight from Amsterdam to New York?",
        }
    ],
)

output = completetion.choices[0].message.content
print(output)

# ------------------------------------------------------------------
# Use OpenAI's function calling feature
# ------------------------------------------------------------------

function_descriptions = [
    {
        "name": "get_flight_info",
        "description": "Get flight information between two locations.",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
            },
            "required": ["loc_origin", "loc_destination"],
        },
    }
]

user_prompt = "What is the next flight from Amsterdam to New York?"

completetion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[{"role": "user", "content": user_prompt}],
    functions=function_descriptions,
    function_call="auto",
)

output = completetion.choices[0].message
print(output)

# ------------------------------------------------------------------
# Add a function
# ------------------------------------------------------------------


def get_flight_info(loc_origin, loc_destination):
    """Get flight information between two locations."""

    flight_info = {
        "loc_origin": loc_origin,
        "loc_destination": loc_destination,
        "datetime": str(datetime.now() + timedelta(hours=2)),
        "airline": "KLM",
        "flight": "KL123",
    }

    return json.dumps(flight_info)


origin = json.loads(output.function_call.arguments).get("loc_origin")
destination = json.loads(output.function_call.arguments).get("loc_destination")
params = json.loads(output.function_call.arguments)

print(origin)
print(destination)
print(params)

chosen_function = eval(output.function_call.name)
flight = chosen_function(**params)

print(flight)

# ------------------------------------------------------------------
# Add function result to prompt for final answer
# ------------------------------------------------------------------

second_completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": user_prompt},
        {"role": "function", "name": output.function_call.name, "content": flight},
    ],
    functions=function_descriptions,
)

response = second_completion.choices[0].message.content
print(response)

# ------------------------------------------------------------------
# Include multiple functions
# ------------------------------------------------------------------

function_descriptions_multiple = [
    {
        "name": "get_flight_info",
        "description": "Get flight information between two locations.",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
            },
            "required": ["loc_origin", "loc_destination"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a flight based on flight information.",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
                "datetime": {
                    "type": "string",
                    "description": "The date and time of the flight, e.g. 2021-07-01T15:00:00",
                },
                "airline": {
                    "type": "string",
                    "description": "The service airline, e.g. Lufthansa",
                },
            },
            "required": ["loc_origin", "loc_destination"],
        },
    },
    {
        "name": "file_complaint",
        "description": "File a complaint as a customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the user, e.g. John Doe",
                },
                "email": {
                    "type": "string",
                    "description": "The email address of the user, e.g. address@mail.com",
                },
                "text": {
                    "type": "string",
                    "description": "Description of the complaint, e.g. The flight was delayed.",
                },
            },
            "required": ["name", "email", "text"],
        },
    },
]

print(function_descriptions_multiple)


def ask_and_reply(prompt):
    """Give LLM a given prompt and get a response."""
    completetion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        functions=function_descriptions_multiple,
        function_call="auto",
    )

    output = completetion.choices[0].message
    return output


# Scenario 1: Ask for flight information
user_prompt = "What is the next flight from Amsterdam to New York?"
print(ask_and_reply(user_prompt))

origin = json.loads(output.function_call.arguments).get("loc_origin")
destination = json.loads(output.function_call.arguments).get("loc_destination")
chosen_function = eval(output.function_call.name)
flight = chosen_function(origin, destination)

print(origin)
print(destination)
print(flight)

flight_datetime = json.loads(flight).get("datetime")
flight_airline = json.loads(flight).get("airline")

print(flight_datetime)
print(flight_airline)

# Scenario 2: Book a new flight
user_prompt = f"I want to book a flight from {origin} to {destination} on {flight_datetime} with {flight_airline}."
print(ask_and_reply(user_prompt))

# Scenario 3: File a complaint
user_prompt = "This is John Doe, I want to file a complaint about my missed flight. It was an unpleasant suprise to find out that my flight was cancelled. Email me a copy of the complaint to john@doe.com."
print(ask_and_reply(user_prompt))

# ------------------------------------------------------------------
# Make it conversational with langchain
# ------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# start conversation with multiple requests
user_prompt = """"
This is Jane Harris. I am an unhappy customer that wants you to do several things.
First, I need to know when the next flight from Amsterdam to New York is.
Please proceed to book that flight for me.
Also, I want to file a complaint about my missed flight. It was an unpleasant suprise to find out that my flight was cancelled.
Email me a copy of the complaint to jharris@email.com.
Please give me confirmation after you have done all of these things.
"""

# Returns the function of the first request
first_response = llm.predict_messages(
    [HumanMessage(content=user_prompt)], functions=function_descriptions_multiple
)
print(first_response)

# Returns the function of the second request
second_response = llm.predict_messages(
    [
        HumanMessage(content=user_prompt),
        AIMessage(content=str(first_response.additional_kwargs)),
        AIMessage(
            role="function",
            additional_kwargs={
                "name": first_response.additional_kwargs["function_call"]["name"],
            },
            content=f"Completed Function {first_response.additional_kwargs['function_call']['name']}",
        ),
    ],
    functions=function_descriptions_multiple,
)

print(second_response)

# Returns the function of the third request
third_response = llm.predict_messages(
    [
        HumanMessage(content=user_prompt),
        AIMessage(content=str(first_response.additional_kwargs)),
        AIMessage(content=str(second_response.additional_kwargs)),
        AIMessage(
            role="function",
            additional_kwargs={
                "name": second_response.additional_kwargs["function_call"]["name"],
            },
            content=f"Completed Function {second_response.additional_kwargs['function_call']['name']}",
        ),
    ],
    functions=function_descriptions_multiple,
)

print(third_response)

# Conversational reply at the end of all requests
fourth_response = llm.predict_messages(
    [
        HumanMessage(content=user_prompt),
        AIMessage(content=str(first_response.additional_kwargs)),
        AIMessage(content=str(second_response.additional_kwargs)),
        AIMessage(content=str(third_response.additional_kwargs)),
        AIMessage(
            role="function",
            additional_kwargs={
                "name": second_response.additional_kwargs["function_call"]["name"],
            },
            content=f"Completed Function {second_response.additional_kwargs['function_call']['name']}",
        ),
    ],
    functions=function_descriptions_multiple,
)

print(fourth_response)
