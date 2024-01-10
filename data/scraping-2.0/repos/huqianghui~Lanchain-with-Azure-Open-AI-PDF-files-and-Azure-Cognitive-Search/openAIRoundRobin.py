import os
import logging
import json
import openai as roundRobinOpenAI
from langchain.chat_models import AzureChatOpenAI

log = logging.getLogger('openAIRoundRobin')
log.setLevel(os.getenv("OPENAI_API_LOGLEVEL").upper())

AZURE_OPENAI_SERVICES=json.loads(os.environ.get("AZURE_OPENAI_SERVICES", "[]"))
AZURE_OPENAI_DEPLOYMENT=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

openAICallCount = 0
openAICount= len(AZURE_OPENAI_SERVICES)

roundRobinOpenAI.api_type="azure"
roundRobinOpenAI.api_version = os.environ.get("OPENAI_API_VERSION"),
roundRobinOpenAI.log= os.getenv("OPENAI_API_LOGLEVEL")

def isRoundRobinMode():
    global openAICount
    log.info(">>>>>>>openAICount: %s" % openAICount)
    return openAICount > 0

def get_openaiByRoundRobinMode():
    global openAICallCount
    global openAICount

    log.info("openAICallCount: %s" % openAICallCount)
    index = openAICallCount % openAICount
    log.info("openAICall index: %s" % index)
    openAICallCount += 1
    roundRobinOpenAI.api_key = AZURE_OPENAI_SERVICES[index].get("OPENAI_API_KEY")
    roundRobinOpenAI.api_base = AZURE_OPENAI_SERVICES[index].get("OPENAI_API_BASE")

    chat = AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        temperature=0)
    return chat

