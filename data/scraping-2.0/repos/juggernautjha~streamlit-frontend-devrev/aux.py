
#! Auxiliary functions :).
import requests
import json
import typing
import langchain
import json
import openai
import os
import pandas as pd

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI


def get_tools(query : str, documentation : typing.List, retriever : typing.Any):
    docs = retriever.get_relevant_documents(query)
    tools = [documentation[d.metadata["index"]]['tool'] for d in docs]
    arguments = [documentation[d.metadata["index"]] for d in docs]
    return tools, arguments

def get_examples(query : str, examples : typing.List, retriever : typing.Any):
    """    Returns:
        _type_: _description_
    """
    docs = retriever.get_relevant_documents(query)
    return docs, [examples[d.metadata["index"]] for d in docs]

def post_request(url: str, query : str, documentation : typing.List, examples : typing.List, open_api_key : str, model_name : str, parse_piro : bool, api_ret : typing.Any, ex_ret : typing.Any):
    """Posts a request to the server

    Args:
        query (str): Query
        documentation (typing.List): API Docs
        examples (typing.List): Examples
        open_api_key (str): Duh
        model_name : Duh
        parse_piro (bool): Whether to use PIRO Prompting
        api_ret, ex_ret : Retriever

    Returns:
        JSON
    """
    url = f"{url}/chat"
    open_api_key = open_api_key
    payload = json.dumps({
        "query": query,
        "tools": [] if not parse_piro else get_tools(query, documentation, api_ret)[-1],
        "examples": [] if not parse_piro else get_examples(query, examples, ex_ret)[-1],
        "use_piro": parse_piro,
        "model_name": model_name,
        "model_type": "openAI" if model_name[:3] == "gpt" else "ollama",
        "openai_key": open_api_key
    })
    print(model_name)
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Accept': '/',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://docs.devrev.ai/',
        'Origin': 'https://docs.devrev.ai/',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)
