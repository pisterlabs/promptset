#!/usr/bin/env python3
from typing import Any, List, Dict
import openai
import requests
from mysecrets import DATABASE_INTERFACE_BEARER_TOKEN
from mysecrets import OPENAI_API_KEY
import logging

def query_database(query_prompt: str) -> Dict[str, Any]:
    num_chunks = input("Enter Number Embeddings To Retrieve: ")
    """
    Query vector database to retrieve chunk with user's input questions.
    """
    url = "http://0.0.0.0:8000/query"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {DATABASE_INTERFACE_BEARER_TOKEN}",
    }
    data = {"queries": [{"query": query_prompt, "top_k": num_chunks}]}
    
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        # process the result
        print(); print("-" * 40); print()
        print(f"\033[43mTop {num_chunks} chunks from query_prompt ",query_prompt," has response with chunks:\033[0m")
        for query_result in result['results']:
            for embedding in query_result['results']:
                page_index = embedding['text'].rfind("Page")
                print(); print(f"Embedding Id:",embedding['id'],"Score:",embedding['score'],"Doc Page:",embedding['text'][page_index:],"Doc text:",embedding['text'][:50],"...")
        return result
    else:
        raise ValueError(f"Error getting chunks for query_prompt {query_prompt}: {response.status_code} : {response.content}")


def apply_prompt_template(question: str) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """
    prompt = f"""
        By considering above input from me, answer the question: {question}
    """
    return prompt

def call_chatgpt_api(user_question: str, chunks: List[str]) -> Dict[str, Any]:
    """
    Call chatgpt api with user's question and retrieved chunks.
    """
    # Send a request to the GPT-3 API
    messages = list(
        map(lambda chunk: {
            "role": "user",
            "content": chunk
        }, chunks))
    #print("List of chunks:",messages)
    question = apply_prompt_template(user_question)
    messages.append({"role": "user", "content": question})
    print(); print("-" * 40); print()
    print(f"\033[43mSending Request to GPT\033[0m")
    response = openai.ChatCompletion.create(
        #engine="selfgengpt35t0301", #azure openai
        model = "gpt-3.5-turbo-0613",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,  # High temperature leads to a more creative response.
    )
    print(); print("-" * 40); print()
    print(f"\033[43mRequest Sent with Embeddings\033[0m")
    return response


def ask(user_question: str) -> Dict[str, Any]:
    """
    Handle user's questions.
    """
    # Get chunks from database.
    chunks_response = query_database(user_question)
    chunks = []
    for result in chunks_response["results"]:
        for inner_result in result["results"]:
            chunks.append(inner_result["text"])
    
    logging.info("User's questions: %s", user_question)
    logging.info("Retrieved chunks: %s", chunks)
    
    response = call_chatgpt_api(user_question, chunks)
    logging.info("Response: %s", response)
    
    return response["choices"][0]["message"]["content"]