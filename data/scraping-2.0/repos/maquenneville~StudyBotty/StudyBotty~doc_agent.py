# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:18:30 2023

@author: marca
"""


import os
import tiktoken
import configparser
import openai
from openai_pinecone_tools import *


def answer_decision_agent(query, context, answer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Your job is to take a query and context, and an attempted answer to the query.  You then decide if the answer to the query is a satisfactory answer.  You must decide, and must answer using either 'yes' or 'no'.",
        },
        {"role": "user", "content": f"Query: {query}"},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Answer: {answer}"},
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


def construct_prompt(question: str, context: str, separator: str = "\n*"):
    header = """Answer the question as truthfully as possible using the provided context. If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know."\n\nContext:\n"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": header},
    ]

    for c in context:
        messages.append({"role": "user", "content": f"Context: {c}"})

    messages.append({"role": "user", "content": f"Q: {question}\nA:"})

    return messages


def doc_agent(query: str, context: str, model=FAST_CHAT_MODEL):
    messages = construct_prompt(query, context)

    response = generate_response(
        messages,
        temperature=0.5,
        n=1,
        max_tokens=1000,
        frequency_penalty=0,
        model=model,
    )
    return response.strip(" \n")
