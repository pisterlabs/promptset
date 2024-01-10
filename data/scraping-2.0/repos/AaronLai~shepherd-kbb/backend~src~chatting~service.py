from fastapi import UploadFile

from langchain.chains import ConversationChain
from backend.src.utilities.pinecone import PineconeConnector
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import WebBaseLoader
from backend.src.utilities.openai import OpenAi
import json

import tempfile
import shutil
import os

from backend.config import Settings

class ChattingService():
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat_with_namespace(self, settings: Settings, embedding,  namespace , text , history = None):
        
        docSearch = PineconeConnector().getDocSearch(settings.PINECONE_API_KEY , settings.PINECONE_ENVIRONMENT ,embedding, namespace)
        

        # qa = OpenAi(settings.OPENAI_API_KEY).getRetrivelQA(docSearch)
        
        # result = qa.run(text)

        # print(result)
        qa = OpenAi(settings.OPENAI_API_KEY).getConversationalRetrievalChain(docSearch)
        result = qa({"question": text , "chat_history": history },return_only_outputs=True)
        # result = json.loads(result["answer"])
        source_documents = []
        for doc in result["source_documents"]:
            source= doc.metadata["source"]
            if source not in source_documents:
                source_documents.append(doc.metadata["source"])
        
        print(source_documents)
        answer = {
            "text":result["answer"],
            "source":source_documents
        }
        
        return answer

