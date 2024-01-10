import os
import dotenv
import openai
from services.embeddings import search_on_token_index
from services.coinmarketcap import read_token_metadata, read_token_quote
from databases.pg import get_prices
from databases.pg import get_prices
from datetime import datetime, date, timedelta
from services.prices import prophet_prices_prediction

dotenv.load_dotenv()

GPT_MODEL = "gpt-3.5-turbo-16k-0613"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def get_prices_for_token(token: str, token_name: str, contents: str):
    quote = read_token_quote(token, "")
    jsonprices = f"""Percent change in the las 7 days: ${quote['quote']['USD']['percent_change_7d']},
    Percent change in the las 30 days: %{quote['quote']['USD']['percent_change_30d']},
    Percent change in the las 60 days: %{quote['quote']['USD']['percent_change_60d']},
    Percent change in the las 90 days: %{quote['quote']['USD']['percent_change_90d']}.\n"""

    precios = []

    prices = get_prices(token)
    for price in prices:
        date_diff = date.today() - price[6].date()
        jsonprices += f"""{date_diff.days} days ago the price was ${price[2]} and the market cap was ${price[4]} (the price went {price[5]}).\n"""
        precios.append(price[2])


    pricesValue = []
    for price in prices:
        pricesValue.append([price[6], price[2]])

    predictions = prophet_prices_prediction(pricesValue, 7)
    prediction = ""
    today = datetime.today()
    day = 1

    for prediction in predictions:
        prediction += f"The prediction of the price for {today + timedelta(days=day)} is ${prediction[2]:.8f}.\n"
        day += 1

    function_description = f"""
    You are an intelligent assistant expert in financial information about the token {token_name} from the Solana Network.
    Always you must give your opinion to the user, and you must answer in the language of the query.
    Function used for performing a financial information of a Solana token named: {token_name}\n\n
    The user will make a query, and you should infer the necessary parameters
    to conduct an efficient response for a financial question.
    Today is {date.today()}.
    The financial information is provided below:
    {jsonprices}
    Add to the response the fact that the prediction for the price for the next 7 days are:
    {prediction}
    Do analysis of the financial information and give summary to the user based in the facts.
    """

    return function_description

def semantic_search_and_completion(question: str, query: str, token: str, user: str):
    """
    Do a semantic search over the document
    Return a list of chunks with the related results
    """

    meta = read_token_metadata(token, "")

    chunks = search_on_token_index(token, query)

    with open(f"data/tokens/{token}/openai_summary.txt") as f:
        contents = f.read()

    # Create completion
    sources = ""
    for chunk in chunks["results"]:
        sources += f"""
        \"\"\"\n{chunk["doc"]}\n\"\"\"\n
        """

    system_prompt = f"""
    You are an intelligent assistant that responds based on the content of a
    documentation about a Solana Token called {meta['name']}. 
    Your answers should be relevant to the content of the documentation
    and address the user's query.

    The user will attach a query, followed by relevant text chunks extracted
    from a semantic search on a documentation transcription.

    Below, delimited by triple quotes (\"\"\"), is a sample structure of the query:

    \"\"\"
    Query: {{ user's query }}

    Sources:
    {{ source text chunk }}
    \n\n
    {{ source text chunk }}
    \n\n
    {{ source text chunk }}
    \"\"\"

    Your goal is to answer the query using the provided text chunks as a source,
    if there is not enough information to resolve the query, do not answer.
    You must answer in the language of the query.

    Next, delimited by triple quotes (\"\"\"), the token name and
    summary will be provided:

    \"\"\"
    Token Name: {meta['name']}.
    \n\n
    Summary: {contents}
    \"\"\"
    """

    user_prompt = f"""
    Query: \"\"\"{question}\"\"\"
    \n\n
    Sources:
    {sources}
    """

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    completion = openai.ChatCompletion.create(
        model=GPT_MODEL, messages=messages, stream=True)

    return {
        "completion": completion,
        "clean_sources": sources
    }


def chat_completion__with_function_calling(user: str, messages: list, token: str):
    """
    Send a request to the OpenAI API to generate a chat completion
    using function calling if is necessary
    """
    meta = read_token_metadata(token, "")

    with open(f"data/tokens/{token}/openai_summary.txt") as f:
        contents = f.read()

    if messages[0]['role'] != "system":
        system_prompt = f"""
        You are an intelligent assistant that responds questions about the token {meta['name']} from the Solana Network. 
        Your answers should be relevant to the content of the documentation and address the user's query.
        The user's questions always will be related to the token {meta['name']} from the Solana Network.
        The summary of the token {meta['name']} is:
        {contents}
        """
        messages.insert(0, {"role": "system", "content": system_prompt})

    function_description = f"""
    Use this function to answer user questions about everything about the token except financial information.
    """

    functions = [
        {
            "name": "token_information_extraction",
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User's question to be answered, based on relevant source text chunks",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query for conducting the semantic search and obtaining relevant source text chunks",
                    },
                },
                "required": ["question", "query"],
            },
        },
        {
            "name": "get_prices_for_token",
            "description": "Get the current price, volume movements, market cap and a list of prices by date for a token",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "User's question to be answered, based the financial data of the token. Prices, market cap, economy direction.",
                    }
                },
                "required": ["question"],
            },
        }
    ]

    completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.2,
    )

    if 'choices' in completion and completion.choices[0].message.get("function_call"):
        fc = completion.choices[0].message.get("function_call")
        if fc["name"] == "get_prices_for_token":
            function_response = get_prices_for_token(token, meta['name'], contents)
            messages.append(
                {
                    "role": "function",
                    "name": "get_prices_for_token",
                    "content": function_response,
                }
            )
            completion = openai.ChatCompletion.create(
            model=GPT_MODEL, messages=messages, stream=True)
            return {
                "completion": completion
            }
        else:
            function_call = completion.choices[0].message.function_call
            query = eval(function_call["arguments"])["query"]
            question = eval(function_call["arguments"])["question"]
            semantic_res = semantic_search_and_completion(question,
                                                    query, token, "")
            return semantic_res
    return {"message": completion.choices[0].message.content}
