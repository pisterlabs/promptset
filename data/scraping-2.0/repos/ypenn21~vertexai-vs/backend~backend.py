import os
from typing import Any, Dict, List
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI #ChatGooglePalm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from backend.customembeddings import CustomVertexAIEmbeddings

def run_g_llm(documents:str, query: str, chat_history: List[Dict[str, Any]] = []):
    #embeddings = VertexAIEmbeddings()  # Dimention 768
    llm = VertexAI(
      model_name='text-bison@001',
      max_output_tokens=256,
      temperature=0.1,
      top_p=0.8,
      top_k=40,
      verbose=True,
    )
    chat = ChatVertexAI()
    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH =5
    embeddings = CustomVertexAIEmbeddings(
       requests_per_minute=EMBEDDING_QPM,
       num_instances_per_batch=EMBEDDING_NUM_BATCH,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts=text_splitter.split_text(documents)
    #texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_texts(texts, embeddings)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vectorstore.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})