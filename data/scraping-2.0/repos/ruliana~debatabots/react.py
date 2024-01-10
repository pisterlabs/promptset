#!/usr/bin/env python3

import os
import json
import openai
import requests
import sys
from hashlib import md5

from joblib import Memory
from pprint import pprint
from html2text import html2text
import chromadb

from lib.chat import *
from lib.markdown_splitter import markdown_splitter

MIN_CHUNK_SIZE = 300

memory = Memory('cache', verbose=0)

# Let's simplify it for now. Just one collection, no persistence.
storage = chromadb.Client()
storage = storage.create_collection('chatbot')


def debug_display(stuff: Any) -> None:
    print("=" * 80)
    pprint(stuff)

def ask_assistant(bot: Bot, function_call_limit=5) -> Bot:
    # function_call_limit avoids infinite loops
    if function_call_limit == 0:
        bot = add_message(bot, system_message('It seems you are stuck. Inform the user what you have tried so far.'))
        return ask_assistant(bot)

    match responder(bot):
        case {'function': 'ask_google', 'arguments': {'search': search}}:
            response = ask_google(search)
            bot = add_message(bot, function_answer('ask_google', response))
            return ask_assistant(bot, function_call_limit - 1)
        case {'function': 'scrap_web_page', 'arguments': {'url': url, 'similar_text': similar_text}}:
            results = query_storage(url=url, query=similar_text)

            # If we don't have the results, scrap the page
            if len(results) == 0:
                page = scrap_web_page(url)
                load_page_to_chromadb(url, page)
                results = query_storage(url=url, query=similar_text)

            bot = add_message(bot, function_answer('scrap_web_page', results))
            return ask_assistant(bot, function_call_limit - 1)
        case {'content': content}:
            return add_message(bot, assistant_message(content))

@memory.cache
def ask_google(query: str) -> str:
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            'q': query,
            'key': os.environ['GOOGLE_SEARCH_KEY'],
            'cx': os.environ['GOOGLE_SEARCH_APP'],
            'excludeSites': 'youtube.com,twitter.com',
        },
    )
    response.raise_for_status()
    result = response.json()

    # Extract fields of interest. GPT can't handle too many tokens.
    entries_of_interest = [
        {'title': item['title'], 'snippet': item['snippet'], 'link': item['link']}
        for item in result['items']
    ]
    entries_of_interest.append({'query': query})
    return json.dumps(entries_of_interest)

@memory.cache
def scrap_web_page(url: str) -> str:
    return requests.get(url).text

def load_page_to_chromadb(url: str, page: str) -> None:
    def generate_id(text):
        hasher = md5()
        hasher.update(text.encode('utf-8'))
        return hasher.hexdigest()

    markdown_text = html2text(page)
    chunks = markdown_splitter(markdown_text)
    # Remove duplicates and empty chunks
    chunks = {
        generate_id(chunk): chunk
        for chunk in chunks
        if len(chunk) >= MIN_CHUNK_SIZE
    }

    for id, chunk in chunks.items():
        try:
            storage.add(
                documents=chunk,
                metadatas={'url': url},
                ids=id,
            )
        except chromadb.errors.IDAlreadyExistsError:
            # Ignore duplicates
            pass

def query_storage(url: str, query: str) -> str:
    result = storage.query(
        query_texts=[query],
        n_results=5,
        where={'url': url},
    )
    docs = result['documents'][0]
    return docs and '\n'.join(docs)

def create_responder_parser(responder: Callable[[Bot], dict[str, str]]) -> Callable[[Bot], dict[str, str]]:
    def wrapper(bot: Bot) -> dict[str, str]:
        match responder(bot):
            case {'function_call': {'name': function_name, 'arguments': arguments}}:
                return {
                    'function': function_name,
                    'arguments': json.loads(arguments),
                }
            case {'content': content}:
                return {'content': content}

    return wrapper

functions = [
    {
        'name': 'ask_google',
        'description': "Knowledge base lookup for things you don't know or are more recent than your cutoff training date",
        'parameters': {
            'type': 'object',
            'properties': {
                'search': {
                    'type': 'string',
                    'description': 'The search question to look up on Google',
                },
            },
            'required': ['search'],
        },
    },
    {
        'name': 'scrap_web_page',
        'description': "Find information inside a website. Use when you don't know the answer, but you know a website that has it." ,
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'The URL of the web page that might have the answer.',
                },
                'similar_text': {
                    'type': 'string',
                    'description': 'The question you are asking. This is used to find the answer in the web page.',
                }
            },
            'required': ['url', 'similar_text'],
        },
    }
]

# Create robust responder
responder = create_react_responder(functions)
responder = create_responder_parser(responder)
responder = with_retries(responder, exception_class=ValueError)
responder = with_retries(responder, exception_class=openai.error.RateLimitError, sleep_time=10)
responder = with_retries(responder, exception_class=openai.error.ServiceUnavailableError, sleep_time=10)

# Main
bot = Bot(
    name='Assistant',
    chat=[
        system_message('Always try to confirm your answer with secondary resources, i.e. scrap more than one web page. Give url references to the user.'),
        # user_message('What was the score in the latest Lakers game?'),
        user_message('How was the weather in Berlin last yesterday?'),
        # user_message('What can you tell me about Kosar Jaff who works at Shopify?')
        # user_message('What are the most recent prompt engineering techniques from 2022 and onwards?')
    ],
)
try:
    bot = ask_assistant(bot)
finally:
    for entry in bot.chat:
        print('=' * 80)
        print(f"{entry['role']}:\n{entry['content']}")
