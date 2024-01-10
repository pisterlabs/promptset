import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Clarifai

from sentence_transformers import SentenceTransformer


def load_embeddings():
    embeddings = SentenceTransformer(model_name_or_path="BAAI/llm-embedder")

    return embeddings


def load_model_gpt3_5_turbo(retriever, memory):
    openai_api_key = os.getenv("OPENAI_API_KEY")

    temperature = st.sidebar.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        key="temperature",

    )

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        temperature=temperature,
        streaming=True,
    )

    qa_chain_gpt3_5_turbo = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    return qa_chain_gpt3_5_turbo


def load_model_mistral(retriever, memory):
    llm = Clarifai(
        pat=os.getenv("CLARIFAI_PAT"),
        user_id=os.getenv("USER_ID"),
        app_id=os.getenv("APP_ID"),
        model_id=os.getenv("MODEL_ID"),
        model_version_id=os.getenv("MODEL_VERSION_ID"),
    )

    qa_chain_mistral = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    return qa_chain_mistral


def load_model_falcon(retriever, memory):
    llm = Clarifai(
        pat=os.getenv("CLARIFAI_PAT"),
        user_id=os.getenv("USER_ID"),
        app_id=os.getenv("APP_ID"),
        model_id=os.getenv("MODEL_ID"),
        model_version_id=os.getenv("MODEL_VERSION_ID"),
    )

    qa_chain_falcon = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    return qa_chain_falcon


def load_model_llama(retriever, memory):
    llm = Clarifai(
        pat=os.getenv("CLARIFAI_PAT"),
        user_id=os.getenv("USER_ID"),
        app_id=os.getenv("APP_ID"),
        model_id=os.getenv("MODEL_ID"),
        model_version_id=os.getenv("MODEL_VERSION_ID"),
    )

    qa_chain_llama = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    return qa_chain_llama
