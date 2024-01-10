import os 
import sys 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS



def run_llm(query:str,faiss_vector_path = None,embedding = None):
    vector_store = FAISS.load_local(faiss_vector_path,embeddings=embedding)

    chat = ChatOpenAI(verbose=True,temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat,chain_type='stuff',retriever=vector_store.as_retriever(),return_source_documents=True)

    return qa({'query':query})


