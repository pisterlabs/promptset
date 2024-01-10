""" Chat services utilities """
import os
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import HTTPException
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set OpenAI API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_KEY2")
openai.organization = os.getenv("OPENAI_ORG2")

# Set the initial prompt
INITIAL_PROMPT = [{"role": "system", "content" : """
        You are a PR Prophet, answering questions about public relations
        and marketing as if you were a prophet in the Bible.  It's meant
        to be a light a playful way for pr and marketing professionals to
        get advice about their craft.  Keep your answers brief almost like
        a two paragraph daily devotional."""}]

openai_models = ["gpt-4-0613", "gpt-4-0613", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613"]

def get_chat_response(chat_history):
    """ Get chat response """
    if chat_history:
        chat_history = INITIAL_PROMPT + chat_history
    else:
        chat_history = INITIAL_PROMPT
    # Append the user message to the chat history
    # Iterate through the models
    for model in openai_models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages= chat_history,
                temperature=1,
                max_tokens=250,
            )
            answer = response.choices[0].message.content
            # Return the answer
            return answer
            
            
        except HTTPException as e:
            print(f"OpenAI API error: {e}")
            continue
