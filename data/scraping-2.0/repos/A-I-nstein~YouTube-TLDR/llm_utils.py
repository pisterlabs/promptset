import os
import math
import streamlit as st
# from palm_api import PALM
from langchain.llms import OpenAI
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
WEAVIATE_URL = os.getenv('WEAVIATE_URL')

@st.cache_resource
def save_embeddings(captions):

    texts = []
    metadata = []

    for record in captions:
        texts.append(record['text'])
        metadata.append({'start': record['start']})

    embeddings = OpenAIEmbeddings()
    weaviate = Weaviate.from_texts(
        texts,
        embeddings,
        metadatas = metadata,
        weaviate_url = WEAVIATE_URL
    )
    return weaviate


def parseNumber(text):
    newText = ""
    text = text.replace('\n', ' ')
    for i in text:
        if (i >= '0' and i <= '9') or i == '.':
            newText += i
    return math.floor(float(newText))

def llm_summary(subtitles):
    template = """Subtitles are enclosed in ###. Summarize the subtitles.
    ###
    {srt}
    ###
    Answer:
    """
    try:
        prompt = PromptTemplate(template=template, input_variables=["srt"])
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)
        # llm = PALM()
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        summary = llm_chain.run(subtitles)
    except Exception as e:
        return 'fail', 'Token limit exceeded.'
    else:
        return 'success', summary

def llm_answer(question, captions):
    try:
        weaviate = save_embeddings(captions)
        docs = weaviate.similarity_search(question, k=1)
        data = docs[0].page_content
    except Exception as e:
        return 'fail', 'Could not extract transcript. Please try a different video.'
    template = """\
    Answer a question when the question and the relevant data is given.\
    Relevant data: {data}\
    Question: {question}\
    Answer:
    """
    try:
        prompt = PromptTemplate(template=template, input_variables=["data", "question"])
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-16k-0613')
        # llm = PALM()
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run({'data':data, 'question':question})
        timestamp = int(docs[0].metadata['start'])
    except Exception as e:
        return 'fail', 'Token limit exceeded.'
    return 'success', (output, timestamp)
