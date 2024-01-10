import os
from typing import Any, Dict, List

from langchain.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceEmbeddings,
    GPT4AllEmbeddings,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.vectorstores import FAISS
import pinecone

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )

    return qa({"question": query, "chat_history": chat_history})


def run_llm_OPENAI(
    query: str,
    search_type=None,
    chat_history: List[Dict[str, Any]] = [],
    chunk_size=None,
    chunk_overlap=None,
    search_kwargs=None,
    chain_type=None,
):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    new_vectorestore = FAISS.load_local(
        f"faiss_index_react/OpenAIEmbeddings/{chunk_size}_{chunk_overlap}", embeddings
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=new_vectorestore.as_retriever(
            search_type=search_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            search_kwargs=search_kwargs,
        ),
        chain_type=chain_type,
        return_source_documents=True,
        verbose=True,
    )

    return qa({"question": query, "chat_history": chat_history})



