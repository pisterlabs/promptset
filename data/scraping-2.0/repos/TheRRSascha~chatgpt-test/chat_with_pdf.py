"""
Using the created ChromaDB, we can facilitate conversations with ChatGPT. The process involves creating an
embedding from the user-provided question. This embedding is then used to retrieve similar text chunks from the
database, which serve as the context for generating a response to the user's question
"""

import os
import chromadb
import tiktoken
from chromadb.config import Settings
import requests
import json
import openai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Set up the encoding for the OpenAI model
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")

# Set your API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Create the embedding function for OpenAI's text-embedding-ada-002 model
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)

# Create a ChromaDB client and collection
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb/",
    anonymized_telemetry=False
))

# Get the collection for querying text chunks
collection = client.get_collection(name="chunk_bookmark", embedding_function=openai_ef)

# Create an OpenAI client
openai_client = openai


def get_query_embedding(query):
    """
    Get the embedding for a query using OpenAI's API.

    Args:
        query (str): The query text.

    Returns:
        list: The embedding of the query.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }

    data = {
        'input': query,
        'model': 'text-embedding-ada-002',
    }

    response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, data=json.dumps(data))
    json_response = response.json()
    response.close()

    if 'data' in json_response:
        embedding = json_response['data'][0]['embedding']
        return embedding
    elif 'error' in json_response:
        error_message = json_response['error'].get('message', 'Unknown error occurred')
        print(f"Error: {error_message}")
    else:
        print("Unknown response format")

    return None


def perform_chromadb_query(embedding):
    """
    Perform a query on the ChromaDB collection using the query embedding.

    Args:
        embedding (list): The query embedding.

    Returns:
        dict: The query results.
    """
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return results


def perform_chat_completion(user_question, document):
    """
    Perform chat completion using OpenAI's Chat API.

    Args:
        user_question (str): The user's question.
        document (str): The document to provide context for the question.

    Returns:
        str: The generated answer.
    """
    try:
        chat = openai_client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.55,
            messages=[
                {"role": "user", "content": user_question},
                {"role": "user", "content": 'Answer question with the text below. \n"""\n' + document + '\n"""'}
            ]
        )
        return chat
    except openai.error.APIError as e:
        return e.error.get('message')
    except openai.error.RateLimitError as e:
        return e.error.get('message')


while True:
    """
    Logic to enable Chatting
    """
    user_input = input("Enter your question (or press 'Enter' to quit): ")
    print("*" * 90)
    if user_input.lower().replace(" ", "") == "":
        break

    question_embedding = get_query_embedding(user_input)
    if question_embedding is None:
        continue

    query_results = perform_chromadb_query(question_embedding)

    if query_results:
        document_text = query_results["documents"][0][0] + query_results["documents"][0][1]

        chat_completion = perform_chat_completion(user_input, document_text)
        # noinspection PyTypeChecker
        number = chat_completion['usage']['total_tokens']
        # noinspection PyTypeChecker
        price = round((number * 0.002) / 1000, 4)
        print(document_text)
        print("*" * 90)
        # noinspection PyTypeChecker
        print(chat_completion['choices'][0]['message']['content'])
        print("*" * 90)
        print(f"Tokens used: {number}, which costs around {price} Dollar")
        print("*" * 90)
print("Exiting....")
