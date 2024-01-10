from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

import os
import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='ap-northeast-1'
)

modelId = 'anthropic.claude-v2:1' 
accept = 'application/json'
contentType = 'application/json'

def chatcompletion(userMessage:str) -> str:
  

    # 推論実行
    body = json.dumps({
        "prompt": '\n\nHuman:{0}\n\nAssistant:'.format(userMessage),
        "max_tokens_to_sample": 500,
    })

    response = bedrock_runtime.invoke_model(
    	modelId=modelId,
    	accept=accept,
    	contentType=contentType,
        body=body
    )
    response_body = json.loads(response.get('body').read())
    return response_body["completion"]

