from langchain.llms import OpenAI, AI21
from langchain.chat_models import ChatOpenAI, PromptLayerChatOpenAI
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
def load_my_env():
    env_path = os.path.dirname(__file__)
    load_dotenv(f'{env_path}/../.streamlit/.env')

## TRACE

def trace_openai(session: str) -> OpenAI:
    enable_tracing(session)
    return get_openai_model()

def trace_ai21(session: str = "vechai", max_tokens = 1000) -> AI21:
    enable_tracing(session)
    return get_ai21_model(model_name="j2-ultra",max_tokens = max_tokens)

def trace_chat_openai(session: str) -> ChatOpenAI:
    enable_tracing(session)
    return get_chat_openai()

## CHAT MODEL
def get_chat_openai(model_name: str = 'text-davinci-003' ,max_tokens: int = 256) -> ChatOpenAI:
    load_my_env()
    #ai_pass = os.getenv("OPENAI")
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI']
    model = ChatOpenAI(model_name=model_name, max_tokens=max_tokens,verbose=True, temperature=0.0)
    print("CHAT OPENAI ready")
    return model

## MODELS

def get_openai_embeddings():
    load_my_env()
    ai_pass = os.getenv("OPENAI")
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI']
    emb = OpenAIEmbeddings()
    print("OPEN AI Embedding ready")
    return emb

def get_openai_model(model_name: str = 'text-davinci-003' ,max_tokens: int = 256) -> OpenAI:
    load_my_env()
    ai_pass = os.getenv("OPENAI")
    os.environ['OPENAI_API_KEY'] = ai_pass
    model = OpenAI(model_name=model_name, max_tokens=max_tokens,verbose=True, temperature=0.0)
    print("OPENAI ready")
    return model

def get_ai21_model(model_name: str = 'j2-jumbo-instruct', max_tokens: int = 256) -> AI21:
    load_my_env()
    ai_pass = st.secrets['AI21']
    model = AI21(ai21_api_key=ai_pass, model=model_name, maxTokens=max_tokens, temperature=0.0)
    print("AI21 ready")
    return model

## TRACING

def enable_tracing(session:str='test-deploy') -> bool:
    load_my_env()
    #lang_key = os.getenv("LANGCHAIN")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN']
    os.environ["LANGCHAIN_SESSION"] = session
    print(f"Enable tracing at {session}")
    return True

