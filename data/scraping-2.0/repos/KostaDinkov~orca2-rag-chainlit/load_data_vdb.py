from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

import os

DATA_PATH="data/"
DB_PATH = "vectorstores/db/"

def create_vector_db():
    # зареждаме наличните в папката data pdf документи
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Processed {len(documents)} pdf files")

    # 'нарязваме' документите на парчета
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)

    # създаваме 'вграждания' (embeddings) и ги запазваме във векторната база данни
    vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(), 
                                        persist_directory=DB_PATH)      
    vectorstore.persist()

if __name__=="__main__":
    create_vector_db()