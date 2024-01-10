"""Chat Utils"""
import asyncio
from typing import Awaitable
from urllib.parse import urljoin, urlparse

import openai
import requests
from bs4 import BeautifulSoup
from langchain import PromptTemplate


def get_chat_history(inputs: tuple) -> str:
    """Formats the chat history into a readable format for the chatbot"""
    res = []
    for human, assistant in inputs:
        res.append(f"Human: {human}\nAI: {assistant}")
    return "\n".join(res)

def get_system_template(system_message: str) -> PromptTemplate:
    """format the system message into a template for the chatbot to use"""
    prompt_template = f"""{system_message}
---
{{context}}
Human: {{question}}
Assistant: """
    template = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return template

async def wrap_done(fn_name: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn_name
    except asyncio.CancelledError:
        pass
    except openai.error.APIError as error:
        print(f"Caught API error: {error}")
    finally:
        # Signal the aiter to stop.
        event.set()
        
def get_links(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and urlparse(href).netloc == '':
            links.append(urljoin(url, href))
    return links

# Function to match strings with an array of objects
def match_strings(keys: list[str], functions):
    # Initialize array to store output
    output = []

    # Loop through the functions array
    for function in functions:
        # If name property of function matches one of the strings in keys
        if function['name'] in keys:
            # Append the function to the output array
            output.append(function)
    
    # Return the output array
    return output

import json
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)