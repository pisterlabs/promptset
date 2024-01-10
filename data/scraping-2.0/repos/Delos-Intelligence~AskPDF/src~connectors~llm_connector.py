import tiktoken

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from connectors.secrets_manager import OPENAI_SECRETS
from prompt_templates import SYSTEM_TEMPLATE, QUESTION_TEMPLATE
from file_manager import read_and_split

import openai
import time
import asyncio
import sys

openai.api_key = OPENAI_SECRETS["API_KEY"]

EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_SECRETS["API_KEY"])

def create_vectorbase(file_info : dict, TEMPFILE_PATH : str):
    print("Creating vectorbase for user " + file_info.user_id)
    chunks = read_and_split(file_info, TEMPFILE_PATH)
    vectorbase = FAISS.from_documents(chunks, EMBEDDINGS)
    print("Vectorbase created for user " + file_info.user_id)
    return vectorbase

def ask_request(question, chat_buffer, vectorbase): 
    #docs = vectorbase.similarity_search(question, k=5)
    print('Asking question: ' + question)
    docs = vectorbase.similarity_search(question, k=5)
    messages = format_content(question, chat_buffer, docs)
    
    print(messages)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=messages,
        max_tokens=800,
        temperature=0,
        stream=True,  # this time, we set stream=True
    )
    
    answer = ''
    yield "beginning of the stream"
    for event in response: 
        event_text = event['choices'][0]['delta'] # EVENT DELTA RESPONSE
        answer = event_text.get('content', '') # RETRIEVE CONTENT
        print(answer, end='')
        sys.stdout.flush()
        yield (answer+'\n').encode('utf-8')
    yield "end of the stream"

    

def format_content(question, chat_buffer, docs):
    system = SYSTEM_TEMPLATE.format()
    user = QUESTION_TEMPLATE.format(question=question, chat_buffer=chat_buffer, context=docs)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]