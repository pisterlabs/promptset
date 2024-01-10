import os
from dotenv import load_dotenv
import json

import azure.functions as func
import logging

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains import ConversationChain
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage

from azure.identity import DefaultAzureCredential
#from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

import Prompts
import SearchUtil

# Load Environment Variables if .env exists
load_dotenv()

OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2023-05-15")
CHAT_DEPLOYMENT = os.environ.get("OPENAI_CHAT_DEPLOYMENT", "gpt35")
CHAT_TEMPERATURE = float(os.environ.get("OPENAI_CHAT_TEMPERATURE", "0.0"))
CHAT_RESPONSE_MAX_TOKENS = int(os.environ.get("OPENAI_CHAT_RESPONSE_MAX_TOKENS", "100"))

AZURE_SEARCH_SERVICE_ENDPOINT = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
AZURE_SEARCH_INDEX_NAME = os.environ["AZURE_SEARCH_INDEX_NAME"]
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "3"))

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed, 
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the 
# keys for each service
# If you encounter a blocking error during a DefaultAzureCredntial resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
azure_credential = DefaultAzureCredential()

# Initialize LangChain with Azure OpenAI
chatLLM = AzureChatOpenAI(
    deployment_name=CHAT_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION,
    max_tokens=CHAT_RESPONSE_MAX_TOKENS,
    temperature=CHAT_TEMPERATURE,
    verbose=True
)

conversation = ConversationChain(
    llm=chatLLM,
    verbose=True)

# Create the Azure Function app
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="ask", methods=["POST"])
def ask(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('ask HTTP trigger function processed a request.')
    
    request_data = req.get_json()
    query = request_data['query']
    
    searchclient = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=azure_credential)
        #credential=AzureKeyCredential(os.environ["AZURE_SEARCH_SERVICE_KEY"]))        
    
    if not query:
        return func.HttpResponse(
            json.dumps({"error": "query is required"}),
            status_code=400,
            mimetype="application/json"
        )
        
    # STEP 1: Rephrase Search Query
    searchquery = SearchUtil.RephraseQuery(
        question=query,
        chatLLM=chatLLM)
    
    logging.info(f'STEP 1: Rephrased query "{query}" to "{searchquery}"')
    
    # STEP 2: Do the Search
    logging.info('STEP 2: Searching the Azure AI search index')
    
    searchresults, sources = SearchUtil.Search(
        searchquery,
        SEARCH_MAX_RESULTS,
        searchclient)
    
    # STEP 3: Answer the Question
    logging.info('STEP 3: RAG answering the question')
    
    systemprompt = Prompts.CHAT_SYSTEMPROMPT.format(sources=searchresults)
    chatprompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(systemprompt),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    conversation.prompt = chatprompt

    with get_openai_callback() as cb:
        response = conversation.predict(input=query)
        total_tokens = cb.total_tokens

        del searchclient
        del query
        del searchquery
        del searchresults

        # Return response
        return func.HttpResponse(
            json.dumps({
                "response": response,
                "sources": sources,
                "total_tokens": total_tokens
            }),
            mimetype="application/json"
        )

@app.route(".well-known/ai-plugin.json", methods=["GET"])
def get_ai_plugin(req: func.HttpRequest) -> func.HttpResponse:
    with open("./.well-known/ai-plugin.json", "r") as f:
        text = f.read()
        return func.HttpResponse(text, status_code=200, mimetype="text/json")


@app.route("logo.png", methods=["GET"])
def get_logo(req: func.HttpRequest) -> func.HttpResponse:
    file_path = "./logo.png"
    with open(file_path, "rb") as file:
        file_data = file.read()
    
    return func.HttpResponse(file_data, status_code=200, mimetype="image/png")

@app.route("openapi.yaml", methods=["GET"])
def get_openapi(req: func.HttpRequest) -> func.HttpResponse:
    with open("./openapi.yaml", "r") as f:
        text = f.read()
        return func.HttpResponse(text, status_code=200, mimetype="text/yaml")