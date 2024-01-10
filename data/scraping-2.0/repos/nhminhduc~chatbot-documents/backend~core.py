import os
import streamlit as st

from typing import Any, List

from langchain import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.text_processor import parse_file, text_to_docs


def run_llm(
    key: str, query: str, index: FAISS, chat_history: List[tuple[str, Any]] = []
):
    chat = ChatOpenAI(
        openai_api_key=key,
        verbose=True,
        temperature=0,
        model="gpt-4"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=index.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})


def create_docs(uploaded_files: List[UploadedFile]):
    all_pages = []
    for uploaded_file in uploaded_files:
        doc = parse_file(uploaded_file)
        pages = text_to_docs(doc)
        for page in pages:
            if page:
                all_pages.append(page)

    return all_pages
