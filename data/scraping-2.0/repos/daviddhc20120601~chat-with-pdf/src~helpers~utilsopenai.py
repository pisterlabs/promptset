import os
import pprint
from tempfile import NamedTemporaryFile
from typing import Any
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

import psycopg2cffi
from langchain.vectorstores.analyticdb import AnalyticDB

my_openai_api_key = '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'





def get_text():
    """Get text from user"""
    input_text = st.text_input("You: ")
    return input_text


def generate_response(query: str, chain_type: str, retriever: VectorStoreRetriever, open_ai_token) -> dict[str, Any]:

    if query == "who are you":
        result_content = """I am HD C """
        result = {'query': query, 'result': result_content, 'source_documents': [
            Document(
                page_content='alibaba cloud SA team hd C ',
                metadata={'source': '/tmp/tmplvqwt_4h.pdf', 'page': 7}),
            Document(
                page_content='super hero SA HD C',
                metadata={'source': '/tmp/tmplvqwt_4h.pdf', 'page': 9}),

        ]}
        return result
    else:
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key = open_ai_token, model_name="gpt-3.5-turbo-16k"),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )
        result = qa({'query': query})
        pprint.pprint(result)

    return result


def transform_document_into_chunks(document: list[Document]) -> list[Document]:
    """Transform document into chunks of {1000} tokens"""
    splitter = CharacterTextSplitter(
        chunk_size=int(os.environ.get('CHUNK_SIZE', 1000)),
        chunk_overlap=int(os.environ.get('CHUNK_OVERLAP', 0))
    )
    return splitter.split_documents(document)




def transform_chunks_into_embeddings(text: list[Document], k: int , open_ai_token , adbpg_host_input, adbpg_port_input, adbpg_database_input, adbpg_user_input, adbpg_pwd_input) -> VectorStoreRetriever:
    """Transform chunks into embeddings"""
    CONNECTION_STRING = AnalyticDB.connection_string_from_db_params(
        driver=os.environ.get("PG_DRIVER", "psycopg2cffi"),
        host=os.environ.get("PG_HOST", adbpg_host_input),
        port=int(os.environ.get("PG_PORT", adbpg_port_input)),
        database=os.environ.get("PG_DATABASE", adbpg_database_input),
        user=os.environ.get("PG_USER", adbpg_user_input),
        password=os.environ.get("PG_PASSWORD", adbpg_pwd_input),
    )

    embeddings = OpenAIEmbeddings(openai_api_key = open_ai_token)
    db = AnalyticDB.from_documents(text, embeddings, connection_string=CONNECTION_STRING)
    return db.as_retriever(search_type='similarity', search_kwargs={'k': k})

def get_file_path(file) -> str:
    """Obtain the file full path."""
    with NamedTemporaryFile(dir='/tmp/', suffix='.pdf', delete=False) as f:
        f.write(file.getbuffer())
        return f.name


def setup(file: str, number_of_relevant_chunk: int, open_ai_token: str , adbpg_host_input, adbpg_port_input, adbpg_database_input, adbpg_user_input, adbpg_pwd_input) -> VectorStoreRetriever:
    # load the document
    loader = PyPDFLoader(file)
    document = loader.load()
    # transform the document into chunks
    chunks = transform_document_into_chunks(document)
    # transform the chunks into embeddings
    return transform_chunks_into_embeddings(chunks, number_of_relevant_chunk ,open_ai_token,adbpg_host_input, adbpg_port_input, adbpg_database_input, adbpg_user_input, adbpg_pwd_input)
