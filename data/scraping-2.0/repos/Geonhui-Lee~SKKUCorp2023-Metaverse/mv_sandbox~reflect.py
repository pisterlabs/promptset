from django.http import HttpResponse, JsonResponse
from mv_backend.lib.database import Database
from mv_backend.lib.common import CommonChatOpenAI
from mv_backend.settings import OPENAI_API_KEY
import json, openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import numpy as np
from numpy.linalg import norm
from langchain.embeddings import OpenAIEmbeddings
import json, openai
from datetime import datetime
from bson.objectid import ObjectId

db = Database()

openai.api_key = OPENAI_API_KEY
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

chat = CommonChatOpenAI()

embeddings_model = OpenAIEmbeddings()
# every prompts for this code

# generate the query score
# !<INPUT 0>! -- Event/thought statements 
# !<INPUT 1>! -- Count 
generate_query_template = """
{event}

Given only the information above, what are {num} most salient high-level questions we can answer about the subjects grounded in the statements?
1)
"""
generate_query_prompt = PromptTemplate(
    input_variables=["event", "num"], template=generate_query_template
)

generate_query = LLMChain(
    llm=chat,
    prompt=generate_query_prompt
)

# find the insight
# !<INPUT 0>! -- Numbered list of event/thought statements
# !<INPUT 1>! -- target persona name or "the conversation"
generate_insights_template = """
Input:
{event}

What are the {name} high-level insights about customers when ordering pizza that can be inferred from the above statement? (example format: insight (because of 1, 5, 3))
1.
"""
generate_insights_prompt = PromptTemplate(
    input_variables=["name", "event"], template=generate_insights_template
)

generate_insights = LLMChain(
    llm=chat,
    prompt=generate_insights_prompt
)

def call(request):

    # declare for start
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)

    
    # openai_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=body["messages"]
    # )
    
    ### mongoDB user's memory ###
    conversation = Database.get_all_documents(db, "conversations", "user")
    data_num = 0

    all_chat_data = []
    before_chat_data = []
    all_chat_data_node = []
    important = []
    for chat_data in conversation:
        data_num += 1
        before_chat_data.append(chat_data["name"] + ": " + chat_data["memory"])
        important.append(chat_data["important"])
    
    if data_num == 0:
        messages_response = body["messages"] + [
            {
                "role": "assistant",
                "content": "insights: " + "None"
            }
        ]

        return JsonResponse({
            "messages": messages_response
        })
    
    data_num = 0
    all_chat_data_string = ""
    for chat_data in reversed(before_chat_data):
        data_num += 1
        if data_num > 100:
            break
        all_chat_data_node.append("[" + str(data_num) + "]" + chat_data)
        all_chat_data.append(chat_data)
        all_chat_data_string += chat_data + "\n"
    
    # all_chat_data = []
    # all_chat_data_node = []
    # all_chat_data_string = ""
    # # now reflect with 100 message data (we wil generate only one query)
    # data_num = 0
    # for chat_data in reversed(body["messages"]):
    #     data_num += 1
    #     if data_num > 100:
    #         break
    #     all_chat_data_node.append("[" + str(data_num) + "]" + chat_data["role"] + ": " + chat_data["content"])
    #     all_chat_data.append(chat_data["role"] + ": " + chat_data["content"])
    #     all_chat_data_string += chat_data["role"] + ": " + chat_data["content"] + "\n"
    
    
    focal_points = generate_query.run(event = all_chat_data_string, num = "1")
    embedded_query = embeddings_model.embed_query(focal_points)
    embedings = embeddings_model.embed_documents(all_chat_data)

    cosine = np.dot(embedings, embedded_query)/(norm(embedings, axis=1)*norm(embedded_query))
    print(cosine)

    chat_data_score = dict(zip(all_chat_data_node, cosine))
    # retrieve to find 30 chat data with the generated query, embedding vetors, recency
    data_num = 0
    recency = 1
    for score, chat_data in zip(reversed(important), all_chat_data):
        data_num += 1
        if data_num > 100:
            break
        recency *= 0.995
        
        chat_data_score["[" + str(data_num) + "]" + chat_data] += 0.1*score + recency
    
    sorted_dict = sorted(chat_data_score.items(), key = lambda item: item[1], reverse = True)
    print(sorted_dict)

    # find the insights with 30 important data
    important_data_string = ""
    data_num = 0
    for chat_data in sorted_dict:
        data_num += 1
        if data_num > 30:
            break
        important_data_string += chat_data[0] + "\n"
    insights = generate_insights.run(name = "assisstant's 5", event = important_data_string)

    # json output

    # messages_response = body["messages"] + [
    #     {
    #         "role": "assistant",
    #         "content": openai_response_message["content"]
    #     }
    # ]
    
    # return JsonResponse({
    #     "messages": messages_response
    # })

    messages_response = body["messages"] + [
        {
            "role": "assistant",
            "content": "insights: " + insights
        }
    ]

    previous = Database.get_all_documents(db, "Reflects", "user")
    print(previous)
    data_num = 0
    node = 0

    for i in previous:
        data_num += 1
    
    if data_num != 0:
        print(i)
        node = i["node"]
    node += 1
    
    datetimeStr = datetime.now().strftime("%Y-%m-%d")
    document_user = {"_id":ObjectId(),"node":node,"timestamp":datetimeStr,"reflect":insights,"name":"user"}
    print(Database.set_document(db, "Reflects", "user", document_user))

    return JsonResponse({
        "messages": messages_response
    })