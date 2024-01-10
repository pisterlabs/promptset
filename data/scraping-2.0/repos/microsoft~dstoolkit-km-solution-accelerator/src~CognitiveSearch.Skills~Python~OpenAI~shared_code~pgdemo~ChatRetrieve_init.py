# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Credits : https://github.com/Azure-Samples/azure-search-openai-demo

import logging
import azure.functions as func
import json
import os
import openai
import time

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from azure.search.documents import SearchClient 

from .approaches.retrievethenread import RetrieveThenReadApproach
from .approaches.readretrieveread import ReadRetrieveReadApproach
from .approaches.readdecomposeask import ReadDecomposeAsk
from .approaches.chatreadretrieveread import ChatReadRetrieveReadApproach

# Environment Variables
oai_endpoint = os.environ["OPENAI_ENDPOINT"]
oai_key = os.environ["OPENAI_KEY"]
oai_version = os.environ["OPENAI_VERSION"]
oai_engine = os.environ["OPENAI_ENGINE"]

AZURE_OPENAI_GPT_DEPLOYMENT = oai_engine
AZURE_OPENAI_CHATGPT_DEPLOYMENT= oai_engine

openai.api_type = "azure"
openai.api_base = oai_endpoint
openai.api_version = oai_version
openai.api_key = oai_key

AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "search"
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "index"
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY") or "key"

KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or "category"
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed, 
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the 
# keys for each service
# If you encounter a blocking error during a DefaultAzureCredntial resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
azure_credential = DefaultAzureCredential()

search_credential = AzureKeyCredential(AZURE_SEARCH_KEY)
search_client = SearchClient(
    endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
    index_name=AZURE_SEARCH_INDEX,
    credential=search_credential)

# Various approaches to integrate GPT and external knowledge, most applications will use a single one of these patterns
# or some derivative, here we include several for exploration purposes
ask_approaches = {
    "rtr": RetrieveThenReadApproach(search_client, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT),
    "rrr": ReadRetrieveReadApproach(search_client, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT),
    "rda": ReadDecomposeAsk(search_client, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT)
}

# Various approaches to integrate GPT and external knowledge, most applications will use a single one of these patterns
# or some derivative, here we include several for exploration purposes
chat_approaches = {
    "rrr": ChatReadRetrieveReadApproach(search_client, AZURE_OPENAI_CHATGPT_DEPLOYMENT, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT)
}

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info(f'{context.function_name} HTTP trigger function processed a request.')
    if hasattr(context, 'retry_context'):
        logging.info(f'Current retry count: {context.retry_context.retry_count}')
        
        if context.retry_context.retry_count == context.retry_context.max_retry_count:
            logging.info(
                f"Max retries of {context.retry_context.max_retry_count} for "
                f"function {context.function_name} has been reached")

    try:
        body = json.dumps(req.get_json())
    except ValueError:
        return func.HttpResponse(
             "Invalid body",
             status_code=400
        )
    
    if body:
        result = compose_response(req.headers, body)
        return func.HttpResponse(result, mimetype="application/json")
    else:
        return func.HttpResponse(
             "Invalid body",
             status_code=400
        )

def compose_response(headers, json_data):
    values = json.loads(json_data)['values']
    
    # Prepare the Output before the loop
    results = {}
    results["values"] = []

    for value in values:
        output_record = transform_value(headers, value)
        if output_record != None:
            results["values"].append(output_record)

    return json.dumps(results, ensure_ascii=False)

## Perform an operation on a record
def transform_value(headers, record):
    try:
        recordId = record['recordId']
    except AssertionError  as error:
        return None

    # Validate the inputs
    try:
        document = {}
        document['recordId'] = recordId
        document['data'] = {}

        assert ('data' in record), "'data' field is required."
        data = record['data']

        method = data["method"]

        if method == "Ask":
            # Ask
            approach = data["approach"]

            impl = ask_approaches.get(approach)
            if impl:
                r = impl.run(data["question"], data("overrides") or {})
                # Response
                document['data']['ask'] = r


        if method == "Chat":
            # Chat
            approach = data["approach"]

            impl = chat_approaches.get(approach)
            if impl:
                r = impl.run(data["history"], data("overrides") or {})
                # Response
                document['data']['chat'] = r

    except KeyError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "KeyError:" + error.args[0] }   ]       
            })
    except AssertionError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "AssertionError:" + error.args[0] }   ]       
            })
    except SystemError as error:
        return (
            {
            "recordId": recordId,
            "errors": [ { "message": "SystemError:" + error.args[0] }   ]       
            })

    return (document)

def ensure_openai_token():
    global openai_token
    if openai_token.expires_on < int(time.time()) - 60:
        openai_token = azure_credential.get_token("https://cognitiveservices.azure.com/.default")
        openai.api_key = openai_token.token
