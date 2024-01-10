import openai
import json
import os
import pkg_resources
from .api_request import send_api_request
from .utilities import DebugValues, PromptPreprocessor
from datetime import datetime
from . import callables

def llm_handler(behaviour: str, context: str, raw_action: str, response_schema: list[dict[str, str]] = None, call = None):
    """
    Send values provided to OpenAI API for processing, returns response
    """

    if DebugValues.verbose_logging:
        print(f"Began API request to OpenAI at {datetime.now().isoformat()}")
    
    preprocessor = PromptPreprocessor(callables.callables) # Hardcoded preprocessor TODO abstract this.

    # Substitute commands in action for their values
    action = preprocessor.preprocess_prompt(raw_action)

    if response_schema:
        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
            {"role": "system", "content": behaviour},
            {"role": "system", "content": context},
            {"role": "user", "content": action}
            ],
            functions=response_schema,
            function_call=call,
            temperature=0,
            )
    else:
        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
            {"role": "system", "content": behaviour},
            {"role": "system", "content": context},
            {"role": "user", "content": action}
            ],
            temperature=0,
            )
    
    if DebugValues.verbose_logging:
        print(f"Behaviour:\n{behaviour}\n")
        print(f"Action:\n{action}\n")
        print(f"Context:\n{context}\n")

    return completion

def openapi_wrapper(context: str, action: str) -> str:
    """
    Specialised wrapper for converting OpenAPI document + request from user
    into an API call.
    """

    behaviour = "You are a tool that converts OpenAPI documentation and a user request into an API call."
    # with open("openai_function_schemas/api_request_schema.json", 'r') as f:
    #     api_request_schema = json.load(f)

    path = "openai_function_schemas/api_request_schema.json"
    loaded_string = pkg_resources.resource_string(__name__, path)
    api_request_schema = json.loads(loaded_string)

    response_schema = [{"name":"api_request", "parameters":api_request_schema}]
    calls = {"name":"api_request"}

    completion = llm_handler(behaviour, context, action, response_schema, calls)
    result = json.loads(completion.choices[0].message.function_call.arguments)

    if(DebugValues.verbose_logging):
        print(f"\nAPI Parameters from LLM:\n{result}\n")

    api_response = send_api_request(result).content.decode('utf-8')

    if(DebugValues.verbose_logging):
        print(f"\nAPI Response:\n{api_response}\n")
    
    return api_response
    
def data_wrapper(context: str, action: str) -> str:
    """
    Specialised wrapper for manipulating a data structure.
    """

    behaviour = "You are a tool that manipulates the response from an API. Respond with only the manipulated data. Do not add any additional text."
    completion = llm_handler(behaviour, context, action)
    result = completion.choices[0].message.content

    if(DebugValues.verbose_logging):
        print(f"\nData Manipulation from LLM:\n{result}\n")
    return result

