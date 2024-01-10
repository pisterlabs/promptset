# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 00:49:04 2023

@author: marca
"""


from openai_pinecone_tools import *


def table_decision_agent(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Your job is to take csv text, and decide if ChatGPT can answer the query using the context.  You must decide, and must answer using either 'yes' or 'no'.",
        },
        {"role": "user", "content": f"Query: {query}\n\nContext: {context}"},
    ]

    response = generate_response(
        messages, temperature=0.0, n=1, max_tokens=10, frequency_penalty=0
    )
    can_answer = None

    if "yes" in response.lower():
        can_answer = True
    elif "no" in response.lower():
        can_answer = False

    else:
        print("Got confused while deciding how to answer your question, I'm sorry!")
        return

    return can_answer


def table_agent(query, context, model=FAST_CHAT_MODEL):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": """You are my Tabular Data Analyzer.  Using provided comma seperated value data, answer the question as truthfully as possible. If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know." """,
        },
        {"role": "user", "content": f"Comma seperated value text:\n{context}"},
        {"role": "user", "content": f"Q: {query}\nA: "},
    ]

    response = generate_response(
        messages, temperature=0.0, n=1, max_tokens=500, frequency_penalty=0, model=model
    )

    return response
