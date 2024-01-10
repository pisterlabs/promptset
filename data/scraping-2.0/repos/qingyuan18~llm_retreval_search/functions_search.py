import os
import re
import json
import boto3
from botocore.config import Config
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.agents import Tool, AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from typing import Any, Dict, List, Optional,Union
from langchain.utilities import SerpAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.memory import ConversationBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["SERPAPI_API_KEY"]="*********"
search = SerpAPIWrapper()


def get_named_parameter(event, name):
    return next(item for item in event['parameters'] if item['name'] == name)['value']

def search_website(event):
    global search
    user_query = get_named_parameter(event, 'user_query') 
    search_ret = search.run(user_query)
    return search_ret


def lambda_handler(event, context):
    result = ''
    response_code = 200
    action_group = event['actionGroup']
    api_path = event['apiPath']
    
    print ("lambda_handler == > api_path: ",api_path)
    
    if api_path == '/searchWebsite':
        result = search_website(event)
    else:
        response_code = 404
        result = f"Unrecognized api path: {action_group}::{api_path}"

    response_body = {
        'application/json': {
            'body': json.dumps(result)
        }
    }
    session_attributes = event['sessionAttributes']
    prompt_session_attributes = event['promptSessionAttributes']
    
    print ("Event:", event)
    action_response = {
        'actionGroup': event['actionGroup'],
        'apiPath': event['apiPath'],
        # 'httpMethod': event['HTTPMETHOD'], 
        'httpMethod': event['httpMethod'], 
        'httpStatusCode': response_code,
        'responseBody': response_body,
        'sessionAttributes': session_attributes,
        'promptSessionAttributes': prompt_session_attributes
    }

    api_response = {'messageVersion': '1.0', 'response': action_response}
        
    return api_response