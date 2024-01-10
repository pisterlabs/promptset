# -*- coding: utf-8 -*-
import streamlit as st
import chromadb

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.model.prompt_template import RETRIEVAL_TEMPLATE_3
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser

LLM_MODEL_3 = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0.2,
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    # frequency_penalty=0.5,
    # presence_penalty=0.5,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
)

LLM_MODEL_4 = ChatOpenAI(
    model_name="gpt-4-1106-preview",
    temperature=0.2,
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    # frequency_penalty=0.5,
    # presence_penalty=0.5,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
)

RETRIEVAL_PROMPT = PromptTemplate(
    template=RETRIEVAL_TEMPLATE_3,
    input_variables=["context", "question"],
)

EMBEDDING_FUNC = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002",
)

LANGCHAIN_VECTOR_DB = Chroma(
    client=chromadb.PersistentClient(path="database/vectordb"),
    collection_name="test",
    embedding_function=EMBEDDING_FUNC,
)

RETRIEVAL_CHAIN = LLMChain(llm=LLM_MODEL_3, prompt=RETRIEVAL_PROMPT, output_parser=StrOutputParser())
