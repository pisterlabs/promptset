"""
The primary chat utilities
"""
import os
import requests
import streamlit as st
import openai
from dotenv import load_dotenv


# Load the environment variables
load_dotenv()

# Get the OpenAI API key and org key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

# Initialize a connection to the redis st

# Define a class for the chat messages
class ChatMessage:
    # Define the init method
    def __init__(self, content, role):
        self.content = content
        self.role = role

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_message(role, content):
    """ Add a message to the chat history.
     Args:
        role (str): The role of the message.  Should be one of "user", "ai", or "system"
        content (str): The content of the message
    """
    
    # If the role is user, we will add a user message formatted as a HumanMessage
    if role == "user":
        message = {"role": "user", "content": content}
    # If the role is ai, we will add an ai message formatted as an AIMessage
    elif role == "ai":
        message = {"role": "assistant", "content": content}
    # If the role is system, we will add a system message formatted as a SystemMessage
    elif role == "system":
        message = {"role": "system", "content": content}
    # Append the message to the chat history
    st.session_state.chat_history.append(message)

COMBS_NOTES = """
You are Luke Combs, a country singer celebrated for your emotional voice and humble nature.
Your music blends traditional country and rock, with lyrics focusing on love, loss, and life.
You connect with fans through authenticity and write songs that reflect ordinary experiences.
Enjoying simple pleasures like fishing, you value hard work and believe in music's healing power.
"""

def get_artist_response(user_message: str):
    """ Get fans answer from the artist via calling the OpenAI API """
    # Define the messages to send to the API
    messages = [
    {
        "role": "system",
        "content": f'You are Luke Combs, the famous country music singer.  Some notes about you are\
            {COMBS_NOTES}.  You are answering acting as the user\'s "co-writer", giving them guidance and\
            assistance on their songwriting journey. Your chat history so far is {st.session_state.chat_history}.'
    },
    {
        "role": "user",
        "content":f"Hi Luke, thanks for co-writing with me.  Here is my next message {user_message}."
    },
    ]
    # Models to iterate through
    models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo"]

    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages = messages,
                max_tokens=500,
                temperature=0.7,
                frequency_penalty=0.5,
                presence_penalty=0.6,
            )
            answer = response.choices[0].message.content
            # Add the user question and AI answer to the chat history
            add_message("user", user_message)
            # Format the user question and the AI answer into ChatMessage objects
            ai_answer = ChatMessage(answer, "ai")
            # Add the user question and AI answer to the chat history
            add_message(ai_answer.role, ai_answer.content)
            # Return the AI answer
            return ai_answer
        except (requests.exceptions.RequestException, openai.error.APIError):
            print(f"Failed to generate response with model {model}")
            continue