"""Tools for researching information about a topic."""

# Adapted from AI Jason's original code here: https://github.com/JayZeeDesign/microsoft-autogen-experiments/blob/main/content_agent.py

from textwrap import dedent
from typing import Any
import json

import requests
from autogen import (
    AssistantAgent,
    UserProxyAgent,
)
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

from hivemind.toolkit.autogen_support import DEFAULT_CONFIG_LIST as config_list
from hivemind.config import SERPER_API_KEY
from hivemind.toolkit.resource_retrieval import scrape


def search(query: str) -> dict[str, Any]:
    """Search a query on Google."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()


def scrape_and_summarize(url: str, printout: bool = False) -> str | None:
    """Scrape website, and also will summarize the content based on objective if the content is too large, objective is the original objective & task that user give to the agent, url is the url of the website to be scraped"""

    if printout:
        print("Scraping website...")
    text = scrape(url)
    if printout:
        print("CONTENT:", text)
    return summarize(text) if len(text) > 8000 else text


def summarize(content: str) -> str:
    """Summarize scraped content."""
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])

    map_prompt = dedent(
        """
        Write a detailed summary of the following text for collating into a research notes:
        '''
        {text}
        '''
        SUMMARY:
        """
    ).strip()
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )
    output = summary_chain.run(
        input_documents=docs,
    )
    return output


def research(query: str) -> str:
    """Research a query, and return the research report."""
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "google search for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scraping website content based on url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Website url to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list,
    }

    researcher = AssistantAgent(
        name="research_function_assistant",
        system_message="Research information about a given query, collect as much information as possible, and generate detailed research results with technical details with all reference links attached; Add TERMINATE to the end of the research report.",
        llm_config=llm_config_researcher,
    )
    user_proxy = UserProxyAgent(
        name="research_function_user_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        function_map={
            "search": search,
            "scrape": scrape_and_summarize,
        },
    )
    user_proxy.initiate_chat(researcher, message=query)
    user_proxy.stop_reply_at_receive(researcher)  # stops the autoreply
    return user_proxy.last_message()["content"]
