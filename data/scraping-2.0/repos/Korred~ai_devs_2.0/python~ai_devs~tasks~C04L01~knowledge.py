import json
import os

import httpx
import openai
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get API key from environment variables
aidevs_api_key = os.environ.get("AIDEVS_API_KEY")

# Create a client instance
client = AIDevsClient(aidevs_api_key)

# Get a task
task = client.get_task("knowledge")
ic(task.data)


# Define function specifications
functions = [
    {
        "type": "function",
        "function": {
            "name": "returnMiddleExchangeRate",
            "description": "Returns the middle exchange rate of a foreign currency",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency": {
                        "type": "string",
                        "description": "Foreign currency in ISO 4217 format (e.g. USD, EUR, GBP, etc.)",
                    },
                },
            },
            "required": ["currency"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "returnCountryInformation",
            "description": "Returns information about a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "English name of the country in lower case (e.g. spain, france, germany, etc.)",
                    },
                    "information_type": {
                        "type": "string",
                        "description": "Type of information to return (e.g. population, area, capital, etc.)",
                    },
                },
            },
            "required": ["country"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answerGeneralQuestion",
            "description": "Default function to answer general questions. Used when no other function can be used to answer the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Answer to a general question based on your knowledge.",
                    },
                },
            },
            "required": ["answer"],
        },
    },
]


# Figure out which function to use to answer the question
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": task.data["question"]},
    ],
    tools=functions,
    max_tokens=200,
)

ic(response)

# Sometimes the function call is not recognized by the model
# In that case just return the content of the message
if response.choices[0].message.tool_calls:
    function_name = response.choices[0].message.tool_calls[0].function.name
    arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

    if function_name == "returnMiddleExchangeRate":
        rates = httpx.get(
            f"https://api.nbp.pl/api/exchangerates/rates/a/{arguments['currency']}"
        ).json()
        current_rate = rates["rates"][0]["mid"]
        answer = current_rate

    elif function_name == "returnCountryInformation":
        # fetch country data from api
        response = httpx.get(
            f"https://restcountries.com/v3.1/name/{arguments['country']}"
        )
        country_data = response.json()[0]

        if arguments["information_type"] == "population":
            answer = country_data["population"]
        elif arguments["information_type"] == "capital":
            # translate capital to Polish
            capital = country_data["capital"][0]
            answer = (
                openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": f'Translate "{capital}" to Polish.',
                        },
                    ],
                    max_tokens=10,
                )
                .choices[0]
                .message.content
            )

    else:
        answer = arguments["answer"]
else:
    answer = response.choices[0].message.content

ic(answer)

response = client.post_answer(task, answer)
ic(response)
