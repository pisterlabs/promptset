# , SearchIndexRequest, IngestRequest, StructuredDataExtractionRequest,
from fts_v2_client import FTS
from carnivore import CarnivoreClient
from search_client_v2 import SearchServiceV2Client
from dataclasses import asdict
from graph_query import GraphQueryClient
from promptlayer import openai
import aiohttp
import asyncio
import os
import instructor
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from langchain_experimental.tools.python.tool import PythonREPL, PythonAstREPLTool
from langchain.tools import StructuredTool
from tiktoken import encoding_for_model
import neo4j

from collections.abc import Callable, Awaitable
import openai
import promptlayer
promptlayer.api_key = os.environ["PROMPTLAYER_API_KEY"]
promptlayer.api_key = os.environ["PROMPTLAYER_API_KEY"]
openai.api_type = "azure"
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_base = "https://brainchainuseast2.openai.azure.com"
openai.api_version = "2023-08-01-preview"
instructor.patch()


# FTS functions
fts = FTS()


def python_repl(code: str):
    repl = PythonREPL()
    return repl.run(code)


def python_repl_ast(code: str):
    repl = PythonAstREPLTool()
    return repl.run(code)


def terminal(command: str):
    return os.popen(command).read()


def graph_query(query: str):
    gqc = GraphQueryClient()
    return gqc.graph_query(query)


def execute_cypher_query(query: str):
    driver = neo4j.GraphDatabase.driver(
        "bolt://n4j.brainchain.cloud:7687", auth=("neo4j", "password"))
    session = driver.session()
    return session.run(query)


def graph_schema():
    gqc = GraphQueryClient()
    return gqc.get_schema()


def fts_ingest_document(url: str) -> Dict[str, Any]:
    return fts.ingest(url)


def fts_indices() -> Dict[str, Any]:
    return fts.indices()


def fts_search_index(url: str = None, keywords: List[str] = [], results: int = 1) -> Dict[str, Any]:
    return fts.search_index(url=url, keywords=keywords, results=results)


def fts_document_qa(url: str = None, query: str = None, keywords: Optional[List[str]] = [], dynamic_schema: Dict[str, str] = {}, results: int = 1) -> Dict[str, Any]:
    """
    For dynamic_schema for fts_document_qa, it must be {}
    """

    return fts.document_qa(url=url, query=query, keywords=keywords, results=results, dynamic_schema={})


def fts_extract(url: str = None, query: str = None, keywords: Optional[List[str]] = [], dynamic_schema: Dict[str, str] = {"summary": "str"}, results: int = 1) -> Dict[str, Any]:
    """
    For dynamic_schema, you need to specify a dictionary of dynamic schema fields to extract from the document. The keys are names of the fields, and the values are the types of the fields. For example, {'name': 'str', 'age': 'int', 'salary': 'Optional[float]'}. For fts_extract, the sky is the limit. Have fun and be creative with the query and the dynamic_schema. The query must specifically refer to all the fields in the dynamic_schema. For example, {'name': 'str', 'age': 'int', 'salary': 'Optional[float]'} would require a query like 'Extract the name, age, and salary of the person who is the CEO of the company Disney from the document... {document_content}'.
    """
    return fts.extract(url=url, query=query, keywords=keywords, results=results, dynamic_schema=dynamic_schema)


search = SearchServiceV2Client()
carnivore = CarnivoreClient()


def web_search(query: Optional[str] = "", search_query: Optional[str] = ""):
    query = query or search_query
    search_tool = StructuredTool.from_function(
        name="web_search", description="Web Search tool: good for typical web searches, weather reports, and other realtime web search query oriented use cases.", func=search.search)
    return search_tool.invoke(query)


def web_content(url: str):
    return carnivore.content_analyzer(url)


def web_cache(url: str):
    return f"'https://webcache.googleusercontent.com/search?q=cache:{url}&strip=1&vwsrc=0'"


def functions():
    functions_ = [
        {
            "name": "graph_schema",
            "type": "function",
            "description": "Returns the schema of the graph database.",
            "parameters": {
                    "type": "object",
                    "properties": {}
            },
            "returns": {
                "type": "object",
                "description": "A JSON object representing the schema of the graph database.",
            }
        },
        {
            "type": "function",
            "name": "execute_cypher_query",
            "description": "Executes a Cypher query against the graph database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A Cypher query. For example 'MATCH (n:FederalSubagency)-[r]-(m) RETURN n,r,m LIMIT 10' You must always use reasonable limits of no more than 10 per query unless specifically instructed to by the user. "
                    }
                },
                "required": ["query"],
            }
        },
        {
            "type": "function",
            "name": "web_search",
            "description": "Google search results tool: good for typical web searches, weather reports, and other realtime web search query oriented use cases. Can use 'query' or 'search_query'",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "A web search query. For example 'effects of PFAS on the brain' or 'what is the weather in San Francisco'.",
                    },
                    "query": {
                       "type": "string",
                        "description": "A web search query. For example 'effects of PFAS on the brain' or 'what is the weather in San Francisco'.",
                    }
                },
                "required": ["search_query"],
            },
        },
        {
            "type": "function",
            "name": "web_content",
            "description": "Fetches and returns the content of a given web URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Returns the contents of a page. If its too long, it will be loaded into the FTS service and you will be directed to use that tool."
                    }
                },
                "required": ["url"]
            }
        },
        {
            "type": "function",
            "name": "web_cache",
            "description": "Generates a Google Cache link for a given web URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL for which you want to generate a Google Cache link."
                    }
                },
                "required": ["url"]
            }
        },
        {
            "type": "function",
            "name": "terminal",
            "description": "Run commands on zsh terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run on the terminal."
                    }
                },
                "required": ["command"]
            }
        },
        {
            "type": "function",
            "name": "python_repl",
            "description": "Run python code in a python console.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to run in the python console."
                    }
                },
                "required": ["code"]
            }
        },
        {
            "type": "function",
            "name": "python_repl_ast",
            "description": "Run python code in a python console.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to run in the python console."
                    }
                },
                "required": ["code"]
            }
        },
        {
            "type": "function",
            "name": "graph_query",
            "description": "Enables you to query the graph database. If you give it a natural language command like 'List 10 federal subagencies', you will get a set of results from the Knowledge Graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to run on the graph database."
                    }
                },
                "required": ["query"]
            }
        },
        # FTS function entries
        {
            "type": "function",
            "name": "fts_ingest_document",
            "description": "Ingests a document into the FTS service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the document to ingest."
                    }
                },
                "required": ["url"]
            }
        },
        {
            "type": "function",
            "name": "fts_indices",
            "description": "Returns a mapping of URLs to index_names in the FTS service.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "type": "function",
            "name": "fts_search_index",
            "description": "Searches the index for the given keywords. You should only use this function with results max of 5 or else you will exceed the context length.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url to associate with the search."
                    },
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of keywords to search for."
                    },
                    "results": {
                        "type": "integer",
                        "description": "The number of results to return. This should be set to a max of 5."
                    },
                    "chunk_indices": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "A list of chunk indices to search for."
                    },
                },
                "required": ["url", "keywords", "results"]
            }
        },
        {
            "type": "function",
            "name": "fts_document_qa",
            "description": "Lets you perform document question answering on ingested URLs, based on a natural language query (which will get applied to all returned document chunks).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to associate with the QA."
                    },
                    "query": {
                        "type": "string",
                        "description": "The query you are running against the document chunks returned based on the keywords provided for the search."
                    },
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional keywords to refine the search.",
                        "default": []
                    },
                    "dynamic_schema": {
                        "type": "object",
                        "description": "A dictionary of dynamic schema fields to extract from the document. The keys are names of the fields, and the values are the types of the fields. For fts_document_qa, this must always be an empty object, {}",
                        "default": {}

                    }
                },
                "required": ["url", "query", "keywords"]
            }
        },
        {
            "type": "function",
            "name": "fts_extract",
            "description": "Extracts structured data from documents in the index. You must specify a schema (dynamic_schema parameter) to make the most of this tool. This means that you specify a schema for the types of objects you want back, and then it gets wrapped (potentially if you identify multiple items in a chunk of text that are of interest) in a list. For a given text document chunk you need to have a directed set of key values with stuff that you want to extract. If you were doing data extract for a biographic document, you could use: for example, {'name': 'str', 'age': 'int', 'salary': 'Optional[float]'}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to associate with the extraction."
                    },
                    "query": {
                        "type": "string",
                        "description": "The query you are running against the document chunks returned based on the keywords provided for the search. You must include any keys you want to extract for the dynamic schema."
                    },
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Keywords to refine the search.",
                        "default": []
                    },
                    "dynamic_schema": {
                        "type": "object",
                        "description": "A dictionary of dynamic schema fields to extract from the document. The keys are names of the fields, and the values are the types of the fields. For example, {'name': 'str', 'age': 'int', 'salary': 'Optional[float]'}. For fts_extract, the sky is the limit. Have fun and be creative with the query and the dynamic_schema. The query must specifically refer to all the fields in the dynamic_schema. For example, {'name': 'str', 'age': 'int', 'salary': 'Optional[float]'} would require a query like 'Extract the name, age, and salary of the person who is the CEO of the company Disney from the document... {document_content}'.",
                        "default": {"summary": "str"}
                    }
                },
                "required": ["url", "query", "keywords", "dynamic_schema"]
            }
        }
    ]

    return functions_
