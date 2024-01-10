from django.http import HttpResponse, JsonResponse
from mv_backend.lib.database import Database
from mv_backend.settings import OPENAI_API_KEY
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from mv_backend.lib.common import CommonChatOpenAI, gpt_model_name
from mv_backend.api.function.retrieve import *
from mv_backend.api.function.reflect import *
from mv_backend.api.function.cefr import *
from mv_backend.api.function.chat import memory_dict
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json, openai
from datetime import datetime
from bson.objectid import ObjectId

db = Database()

openai.api_key = OPENAI_API_KEY

chat = CommonChatOpenAI()

def call(request):
    global cefr_string
    cefr_string = "A1"

    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)

    user_name = ""
    for chat_data in body["messages"]:
        if chat_data["role"] == "user_name":
            user_name = chat_data["content"]
            break
    
    reflect = Database.get_recent_documents(db, user_name, "Reflects_Kor", 1)
    retrieve = Database.get_recent_documents(db, user_name, "Retrieves_Kor", 1)
    cefr = Database.get_recent_documents(db, user_name, "CEFR_GPT", 1)
    for i in reflect:
        reflect_string = i["reflect"]
    for i in retrieve:
        retrieve_string = i["retrieve"]
    for i in cefr:
        cefr_string = i["cefr"]
    
    messages_response = [
        {
            "role": "reflect",
            "content": reflect_string
        }
    ]

    messages_response += [
        {
            "role": "retrieve",
            "content": retrieve_string
        }
    ]

    messages_response += [
        {
            "role": "cefr",
            "content": cefr_string
        }
    ]

    print(messages_response)
    
    return JsonResponse({
        "messages": messages_response
    })