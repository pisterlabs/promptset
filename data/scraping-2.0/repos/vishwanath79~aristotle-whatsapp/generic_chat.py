import aiohttp
from openai import AsyncOpenAI
import logging
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import os
import openai
from flask import Flask, request
from openai import OpenAI
import sys
from pathlib import Path

#sys.path[0] = str(Path(sys.path[0]).parent)
from cred import OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
client = OpenAI(api_key=OPENAI_API_KEY)


def chat_with_gpt(question, conversation_history, personality):
    """
    Function to interact with ChatGPT.

    Args:
    - question (str): The question to ask ChatGPT.
    - conversation_history (list): List containing conversation history.

    Returns:
    - str: Answer from ChatGPT.
    """

    # Ensure conversation history doesn't exceed the token limit (e.g., 4096 tokens)
    while sum(len(message['content'].split()) for message in conversation_history) > 4096:
        conversation_history.pop(0)  # Remove the oldest message

    # Create a message with the user's question
    user_message = {"role": "user", "content": question}

    # Add the user's message to the conversation history
    conversation_history.append(user_message)

    # Use OpenAI to get an answer
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": personality},
            *conversation_history,
        ]
    )

    # Extract the answer from the response
    answer = response.choices[0].message.content
    print("#######This is the generic  chat response")

    # Add the model's response to the chat windows conversation historyha
    conversation_history.append({"role": "assistant", "content": answer})
    print(answer)

    return answer



