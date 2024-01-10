from langchain.embeddings import GPT4AllEmbeddings
from dataSplitter import split_file_data
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import pandas as pd

def get_vectorstore(chunks, model_name="all-MiniLM-L6-v2"):
    
    print("loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    df = pd.DataFrame(chunks, columns=['page_content','metadata'])

    print("extracting metadata and page content from chunks...")
    texts =  pd.DataFrame(df['page_content'].tolist(), columns=['page_content', 'value'])['value'].tolist()
    metadatas = pd.DataFrame(df['metadata'].tolist(), columns=['metadata', 'value'])['value'].tolist()


    print(f"creating vectorstore for {len(texts)} chunks...")
    return FAISS.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings)




