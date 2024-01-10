import json
import os
from fastapi import UploadFile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from src.api import S3Connector
from src.config import Settings
from src.service.chatbot.chatbot import Chatbot
from celery import Celery
from kombu.serialization import register
from src.api import celeryconfig
from src.service.chatbot.loaders.s3_directory_loader import S3DirectoryLoader

celery_app = Celery(__name__)
celery_app.config_from_object(celeryconfig)


@celery_app.task(name="add_doc_update_index")
def add_doc_update_index(file_info: dict, document_id: str,
                         persist_directory: str,
                         docs_path: str = None) -> dict[str, str]:
    s3connector = S3Connector()
    try:
        file = UploadFile(**file_info)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        s3connector.add_doc(file=file, document_id=document_id)
        loader = S3DirectoryLoader(prefix=docs_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
        db.persist()
        return {"message": f"Index successfully updated!"}
    except Exception as e:
        # eliminate doc from s3 if something went wrong
        s3connector.delete_object(object_id=document_id)
        return {"message": f"Something went wrong: {e}"}
