import config
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def index_query(user_query):
    chatbot = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=config.DevelopmentConfig.OPENAI_KEY,
            temperature=0, model_name="gpt-3.5-turbo"
        ), 
        chain_type="stuff", 
        retriever=FAISS.load_local("schema_docs_old", OpenAIEmbeddings())
            .as_retriever(search_type="similarity", search_kwargs={"k":1})
    )

    return chatbot.run(query=user_query)

def openai_query(user_query):
    messages = [
    {"role": "system", "content" : "Youre a kind helpful assistant"}
    ]

    messages.append({"role": "user", "content": user_query})

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
    )

    try:
        answer = response.choices[0].message.content
    except:
        answer = "Sorry, I don't understand. Can you rephrase your question?"

    return answer

def openai_query_context(user_query, context):
    messages = [
    {"role": "system", "content" : "Youre a kind helpful assistant"}
    ]

    content = context + user_query
    messages.append({"role": "user", "content": content})

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
    )

    try:
        answer = response.choices[0].message.content
    except:
        answer = "Sorry, I don't understand. Can you rephrase your question?"

    return answer

#############################
def story_creation_prompt(user_query):
    story_index_output = index_query("return story schema and example")
    story_openai_output = openai_query_context(story_index_output, user_query + ' (do NOT return the schema NOR any additional text)')

    return story_openai_output

def timeline_creation_prompt(user_query, story_data):
    timeline_index_output = index_query("return timeline schema and example")
    timeline_openai_output = openai_query_context(timeline_index_output, user_query + story_data + ' (do NOT return the schema NOR any additional text)')

    return timeline_openai_output


