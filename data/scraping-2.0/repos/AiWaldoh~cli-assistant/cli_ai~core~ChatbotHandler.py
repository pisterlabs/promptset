import os
import json
from core.ChatGPTBase import ChatGPTBase
from curl_cffi import requests
from bs4 import BeautifulSoup
from newspaper import Article
from requests.models import Response
import nltk
import os
import json
import openai
import urllib.parse

from core.SearchResult import SearchResult


def url_encode_string(s):
    return urllib.parse.quote_plus(s)


SYSTEM_MESSAGE_CONTEXT = """
    You are a genius when it comes to understanding the context of a message.
    1. If the context of the message is telling you to execute a linux command, you will return that command in json format.
    2. If the context is not asking to execute a linux command, for example, asking for help regarding a command, respond as best possible.
    """


SYSTEM_MESSAGE_QA = """You Answer as a QA Chatbot"""

import json


def fetch_google_search_results(query, num_results=20):
    """Fetch Google search results for the given query."""
    query = url_encode_string(query)
    r: Response = requests.get(
        f"https://www.google.com/search?client=firefox-b-d&q={query}&num={num_results}",
        impersonate="chrome101",
    )
    # print("1")
    soup = BeautifulSoup(r.content, "html.parser")
    results = soup.find_all("div", class_="tF2Cxc")
    # print("2")
    # Extract details from the search results
    search_results = []
    for result in results:
        title_element = result.find("h3", class_="LC20lb")
        link_element = result.find("a", href=True)
        description_element = result.find("div", class_="VwiC3b")

        title = title_element.text if title_element else None
        link = link_element["href"] if link_element else None
        description = description_element.text if description_element else None

        # Filtering out links not starting with http or https
        if link and (link.startswith("http://") or link.startswith("https://")):
            search_results.append(SearchResult(title, link, description))

    return search_results


class ChatbotHandler:
    def __init__(self, api_key):
        self.chatbot = ChatGPTBase(api_key)
        self.initialize_functions()  # Initialize custom functions

    def initialize_functions(self):
        # Define the custom functions
        custom_functions = [
            {
                "name": "execute_command",
                "description": "Determine if the message is a command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "User input command.",
                        }
                    },
                },
            },
            {
                "name": "search_google",
                "description": "Search for a query on Google",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The query to search on Google if user specified to search.",
                        }
                    },
                    "required": ["search_query"],
                },
            },
            {
                "name": "load_website",
                "description": "Load a website URL and provide a summary or its content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "website_url": {
                            "type": "string",
                            "description": "The URL of the website to load if user specified to check a website.",
                        }
                    },
                    "required": ["website_url"],
                },
            },
        ]
        # Add them to the base chatbot
        for func in custom_functions:
            self.chatbot.add_custom_function(func)

    def answer_from_context(self, question):
        last_user_message = self.chatbot.history[-2]["content"]
        # print("fds")
        # print(last_user_message)
        # print(self.chatbot.history)
        ai_response_json = self.chatbot.history[-1]["content"]
        # print(ai_response_json)
        # last_ai_reply = ai_response_json["result"]["reply"]
        # print("fds")
        # print(last_ai_reply)
        context_string = f"""
            You are a genius at understanding context and you will answer the Human's question based on the following context:
            Question: {last_user_message} 
            Answer: {ai_response_json}
            
            Answer this follow up question based on the previous interaction's context:
            Question:{question} 
            Answer:"""

        response = self.chatbot.ask_chatbot(
            prompt=context_string,
            system_message=SYSTEM_MESSAGE_QA,
            history=True,
            original_command=question,
            memory_template="""{{"result": {{"message_type": "normal", "reply": "{}"}}}}""",
        )
        return response, True

    def answer_or_execute_command(self, command):
        response_message = self.chatbot.ask_chatbot(
            command, SYSTEM_MESSAGE_CONTEXT, history=True
        )
        searcher = SearchResult("title", "url", "description")
        # Check if a function call was invoked
        if response_message.get("function_call"):
            # This is the part where the command logic will go, as the function has been triggered
            # You can extract relevant information from response_message["function_call"]
            function_name = response_message["function_call"]["name"]

            if function_name == "search_google":
                res = json.loads(response_message["function_call"]["arguments"])
                search_query = res["search_query"]
                print("Searching for:  " + search_query)
                results = fetch_google_search_results(search_query)
                # for result in results:
                #     print(result.title)
                #     print(result.url)
                #     print(result.description)
                # result.fetch_and_parse_article()  # Fetch and parse article content
                # result.display()
                return results, False
            elif function_name == "load_website":
                print("Scraping website...")
                arguments = json.loads(response_message["function_call"]["arguments"])
                # print(arguments)
                website_url = arguments["website_url"]
                print(website_url)
                res = searcher.fetch_and_parse_article(website_url)
                # print(res)
                return res, False
            elif function_name == "execute_command":
                return response_message["function_call"], True
            else:
                return response_message["content"], False
        else:
            return response_message["content"], False
