import redis
from langchain.vectorstores import Redis
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import streamlit as st
import os
REDIS_URL = 'redis://localhost:6379'

redis_store = Redis.from_existing_index(OpenAIEmbeddings(client=None, 
                                                         openai_api_key=os.environ['OPEN_AI_KEY']),index_name='WebMD', redis_url = REDIS_URL)

def get_augmented_prompt(user_input: str) -> str:
    probable_diseases = redis_store.similarity_search(user_input, k = 4)
    prompt =f""" Given the following symptoms {user_input}, which one of the following 5 diseases could the patient be suffering from\n'
        """
    for i, probable_disease in enumerate(probable_diseases):
        prompt += str(i) + '. ' + probable_disease.page_content
    return prompt
def get_gpt_answer(user_input: str) -> str:
    prompt = get_augmented_prompt(user_input)
    messages = [
        SystemMessage(content='You are a helpful assistant that helps diagnose diseases'),
        HumanMessage(content = prompt)
    ]
    chat = ChatOpenAI(temperature=0, model = 'gpt-4', openai_api_key=os.environ['OPEN_AI_KEY'])
    ai_response = chat(messages=messages)
    return ai_response.content
