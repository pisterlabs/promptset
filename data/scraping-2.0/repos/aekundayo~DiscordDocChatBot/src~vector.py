from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, qdrant, weaviate, Redis
import os
import asyncio

vectorpath = 'docs/vectorstore'

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

async def persist_new_chunks(text_chunks):
    historyvectorstore = None
    if is_folder_empty(vectorpath):
      historyvectorstore = FAISS.from_texts(texts=text_chunks, embedding=OpenAIEmbeddings())
    else:
      historyvectorstore = FAISS.load_local(vectorpath , OpenAIEmbeddings() ,"faiss_discord_docs")
    historyvectorstore.add_texts(text_chunks)
    historyvectorstore.save_local(vectorpath, "faiss_discord_docs")

    return historyvectorstore

def get_history_vectorstore():
    if is_folder_empty(vectorpath):
      historyvectorstore = FAISS.from_texts(texts=None, embedding=OpenAIEmbeddings())
    else:
      historyvectorstore = FAISS.load_local(vectorpath , OpenAIEmbeddings() ,"faiss_discord_docs")
    return historyvectorstore