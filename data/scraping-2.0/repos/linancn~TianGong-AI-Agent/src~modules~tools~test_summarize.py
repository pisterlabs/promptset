import streamlit as st

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.xata import XataVectorStore
from xata.client import XataClient


def main_chain():
    langchain_verbose = st.secrets["langchain_verbose"]
    openrouter_api_key = st.secrets["openrouter_api_key"]
    openrouter_api_base = st.secrets["openrouter_api_base"]

    selected_model = "anthropic/claude-2"
    # selected_model = "openai/gpt-3.5-turbo-16k"
    # selected_model = "openai/gpt-4-32k"
    # selected_model = "meta-llama/llama-2-70b-chat"

    llm_chat = ChatOpenAI(
        model_name=selected_model,
        # temperature=0,
        streaming=True,
        verbose=langchain_verbose,
        openai_api_key=openrouter_api_key,
        openai_api_base=openrouter_api_base,
        headers={"HTTP-Referer": "http://localhost"},
        # callbacks=[],
    )

    chain = load_summarize_chain(llm_chat, chain_type="stuff")

    return chain


def fetch_uploaded_docs():
    """Fetch uploaded docs."""
    username = st.session_state["username"]
    session_id = st.session_state["selected_chat_id"]
    client = XataClient()
    query = """SELECT content FROM "tiangong_chunks" WHERE "username" = $1 AND "sessionId" = $2"""
    response = client.sql().query(statement=query, params=(username, session_id))

    return response

def summarize_docs():

    chain = main_chain()

    xata_api_key = st.secrets["xata_api_key"]
    xata_db_url = st.secrets["xata_db_url"]
    embeddings = OpenAIEmbeddings()
    table_name = "tiangong_chunks"

    vector_store = XataVectorStore(
        api_key=xata_api_key,
        db_url=xata_db_url,
        embedding=embeddings,
        table_name=table_name,
    )

    docs = vector_store.similarity_search("material flow analysis", k=80)

    response = chain.run(docs)

    return response
