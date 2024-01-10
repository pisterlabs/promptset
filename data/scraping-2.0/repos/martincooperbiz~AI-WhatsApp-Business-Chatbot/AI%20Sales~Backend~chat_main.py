import streamlit as st
import openai
from streamlit_chat import message
import pinecone
import os
from Backend import prompts

from dotenv import load_dotenv
load_dotenv()

# We will  pasting our openai credentials over here
openai.api_key = os.getenv("OPENAI_API_KEY")

# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

print(pinecone.list_indexes(), os.getenv("OPENAI_API_KEY"))

index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
message_history = []
globalName = ''

# openai embeddings method
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )

    return response['data'][0]['embedding']


def find_top_match(query, k):
    query_em = get_embedding(query)
    result = index.query(query_em, top_k=k, includeMetadata=True)

    return [result['matches'][i]['metadata']['context'] for i in range(k)], [result['matches'][i]['score'] for i in range(k)]


def get_message_history(contexts):
    message_hist = message_history
    message_hist.append([{"role": "user", "content": contexts}])

    return message_hist


def chat(var, message, questionType, role="user"):
    message_history.append({"role": role, "content": f"{var}"})
    print("message request: ", message)
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message
        # messages = [{"role": role, "content": f"{var}"}]
    )
    print("chat: ", completion)
        
    reply = completion.choices[0].message.content
    if questionType.find("BOOOB") != -1: # this is for getting name
        if reply == "BOOOB":
            message_history.append({"role": "assistant", "content": f"{reply}"})

            return reply
        else:
            globalName = "Now I'm talking with" + reply #
            messages = [{"role": "system", "content": prompts.admin_system_message + globalName}]
            messages.append({"role": "user", "content":  var })

            message_history.append({"role": "assistant", "content": f"{reply}"})
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
                # messages = [{"role": role, "content": f"{var}"}]
            )
            reply1 = completion.choices[0].message.content
            return reply1
    else:
        return reply


def get_response(user_input, questionType):
    # context, score = find_top_match(user_input, 1)

    # Generate human prompt template and convert to API message format

    # query_with_context = prompts.human_template.format(query=user_input, context=questionType)
    
    # Convert chat history to a list of messages
    messages = [{"role": "system", "content": prompts.admin_system_message}]
    # if len(message_history) > 2:
    #     messages.append(message_history[:-2])
    messages.append({"role": "user", "content": "'"+ user_input + "'" + questionType})

    response = chat(user_input, messages, questionType)
    return response
