import aiohttp, asyncio, os
import instructor
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from collections.abc import Callable, Awaitable
from promptlayer import openai
import promptlayer
promptlayer.api_key = os.environ["PROMPTLAYER_API_KEY"]

from tools import python_repl, python_repl_ast, terminal
from tools import web_search, web_content, web_cache, functions, graph_query, graph_schema, execute_cypher_query

instructor.patch()
from langchain.tools import StructuredTool
from search_client_v2 import SearchServiceV2Client
search = SearchServiceV2Client()
from carnivore import CarnivoreClient
carnivore = CarnivoreClient()

import json
import promptlayer
from schema import Plan, Step
from promptlayer import openai

SYSTEM_MESSAGE = """
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a guestion.
Respond to the following prompt by using function_call and then summarize your actions, any information from your findings, as well as footnotes and references for any links or information you encounter in your actions. Ask for clarification if a user request is ambiguous.
"""

FUNCTIONS = functions()

# Maximum number of function calls allowed to prevent infinite or lengthy loops
MAX_CALLS = 10

MODELS = {
    "gpt-4-32k": {
        "model": "gpt-4-32k",
        "context-length": 2**15
    },
    "gpt-3.5-turbo-16k": {
        "model": "gpt-3.5-turbo-16k",
        "context-length": 2**14,
    },
    "gpt-4-8k": {
        "context-length": 2**13
    }
}

FAST_MODEL = "gpt-3.5-turbo-16k"
SMART_MODEL = "gpt-4-32k"
MODEL = SMART_MODEL

async def get_preview_response(messages, model="gpt-4-32k", functions=FUNCTIONS):
    import openai
    promptlayer.api_key = os.environ["PROMPTLAYER_API_KEY"]
    openai.api_type = "azure"
    openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
    openai.api_base = "https://brainchainuseast2.openai.azure.com"
    openai.api_version = "2023-08-01-preview"

    v = await openai.ChatCompletion.acreate(
        model=model,
        functions=functions,
        function_call="auto",
        temperature=TEMPERATURE,
        messages=messages
    )
    return v

TEMPERATURE=0.3
async def get_openai_response(messages, model=MODEL, functions=FUNCTIONS):
    openai.api_type = "azure"
    openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
    openai.api_base = "https://brainchainuseast2.openai.azure.com"
    openai.api_version = "2023-08-01-preview"
    return await openai.ChatCompletion.acreate(
        engine=model,
        functions=functions,
        function_call="auto",  # "auto" means the model can pick between generating a message or calling a function.
        temperature=TEMPERATURE,
        messages=messages,
    )


from tools import graph_query, fts_ingest_document, fts_indices, fts_search_index, fts_document_qa, fts_extract

async def process_user_instruction(instruction, prev_steps, functions=FUNCTIONS, model=MODEL):
    num_calls = 0
    messages = [
        {"content": SYSTEM_MESSAGE, "role": "system"},
        {"content": f"Previous steps included this info: {prev_steps}", "role": "assistant"},
        {"content": instruction, "role": "user"},
    ]

    print(f"Calling GPT-4 with {len(messages)} messages")
    print(messages)

    while num_calls < MAX_CALLS:
        if model == "gpt-5":
            response = await get_preview_response(messages, model=model)
        else:
            response = await get_openai_response(messages, model=model)
        message = response["choices"][0]["message"]

        if message.get("function_call"):
            message["content"] = message["function_call"]["name"]

            print(f"\n>> Function call #: {num_calls + 1}\n")
            print(message["function_call"])
            messages.append(message)
            function, args = message["function_call"]["name"], json.loads(message["function_call"]["arguments"])
            print(f"Calling function: {function} with args: {args}")
            available_functions = {
                "graph_schema": graph_schema,
                "execute_cypher_query": execute_cypher_query,
                "web_search": web_search, 
                "web_content": web_content,
                "web_cache": web_cache, 
                "python_repl": python_repl, 
                "python_repl_ast": python_repl_ast, 
                "terminal": terminal,
                "graph_query": graph_query,
                "fts_ingest_document": fts_ingest_document,
                "fts_indices": fts_indices,
                "fts_search_index": fts_search_index,
                "fts_document_qa": fts_document_qa,
                "fts_extract": fts_extract,
            }

            function = available_functions[function]
            response_message = function(*list(args.values()))
            print(response_message)
            # For the sake of this example, we'll simply add a message to simulate success.
            # Normally, you'd want to call the function here, and append the results to messages.
            messages.append({ "role": "assistant", "content": json.dumps(response_message) })

            if "Error" in response_message:
                messages.append({ "role": "function", "content": "failure", "name": str(function.__name__)})
            else:
                messages.append({ "role": "function", "content": "success", "name": str(function.__name__)})
            
            print(messages)
            num_calls += 1
        else:
            print("\n>> Message:\n")
            print(message["content"])
            return message["content"]

    if num_calls >= MAX_CALLS:
        print(f"Reached max chained function calls: {MAX_CALLS}")
    
    return messages

class Capability(BaseModel):
    name: str
    description: str
    func: str

class Capabilities(BaseModel):
    capabilities: List[Capability] = []

from typing import List, Dict, Any, Optional, Union

class Executor:
    def __init__(self, step: Optional[Step] = None, dependency_results: Optional[Dict] = {}, functions: List[Callable] = FUNCTIONS):
        if not step:
            raise Exception("Must provide a plan to execute")
        self.step = step

    async def begin(self, prev_steps: List[Any] = []):
        instruction = self.step.step
        self.prev_results = prev_steps
        return await process_user_instruction(instruction, self.prev_results, functions=FUNCTIONS)
