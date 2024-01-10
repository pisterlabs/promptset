import faiss
import numpy as np
import math, os, random, csv, getpass
from transformers import AutoTokenizer, AutoModel
import torch
from text2vec import SentenceModel
from sentence_transformers import SentenceTransformer
import re
from langchain.vectorstores import FAISS
import faiss
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
import openai
import os
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader, TextLoader
from typing import List
from pathlib import Path
from config import OPEN_API_KEY, PINECONE_KEY, SERP_API_KEY
os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
os.environ["SERPAPI_API_KEY"] = SERP_API_KEY
#OpenAI類默認對應 「text-davinci-003」版本：
#ChatOpenAI類默認是 "gpt-3.5-turbo"版本
#OpenAI是即將被棄用的方法，最好是用ChatOpenAI
## 檢查是否有可用的GPU
#EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#llm_chat = ChatOpenAI(temperature=0) #GPT-3.5-turbo
embeddings = OpenAIEmbeddings()

# 獲取當前 Python 腳本的絕對路徑
current_script_path = Path(__file__).resolve().parent

# 在當前目錄下檢查 "faiss_index" 資料夾是否存在
faiss_index_path = current_script_path / 'faiss_index'
faiss_index_path.mkdir(parents=True, exist_ok=True)

# 檢查 "index.faiss" 文件是否存在
index_path = faiss_index_path / 'Q.faiss'
if not index_path.exists():
    texts = ['這是一個測試文本']
    db = FAISS.from_texts(texts, embeddings)
    db.save_local(folder_path=str(faiss_index_path),index_name="Q")
def process_and_store_documents(file_paths: List[str]) -> None:
    openai.api_key = 'api_key'
    os.environ["OPENAI_API_KEY"] = openai.api_key
    embeddings = OpenAIEmbeddings()
    def init_txt(file_pth: str):
        loader = TextLoader(file_pth)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=180, chunk_overlap=0, separators=["\n"]
        )
        split_docs_txt = text_splitter.split_documents(documents)
        return split_docs_txt
    
    def init_csv(file_pth: str):
        # my_csv_loader = CSVLoader(file_path=f'{file_pth}',encoding="utf-8", 
        #                           csv_args={'delimiter': ','
        # })
        loader = CSVLoader(f'{file_pth}', encoding = "utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
        )
        split_docs_csv = text_splitter.split_documents(documents)
        return split_docs_csv
    

    doc_chunks = []
    for file_path in file_paths:
        extension = os.path.splitext(file_path)[-1].lower()  # Get the file extension
        if extension == '.txt':
            txt_docs = init_txt(file_path)
            doc_chunks.extend(txt_docs)
        elif extension == '.csv':
            csv_docs = init_csv(file_path)
            doc_chunks.extend(csv_docs)
    current_script_path = Path(__file__).resolve().parent
    # 在當前目錄下找 "faiss_index" 資料夾
    faiss_index_path = current_script_path / 'faiss_index'
    # 加載FAISS
    docsearch = FAISS.load_local(str(faiss_index_path), embeddings, index_name="Q")
    new_metadatas = [doc.metadata for doc in doc_chunks]
    # 提取每個文檔的實際來源
    docsearch.add_texts([t.page_content for t in doc_chunks], metadatas=new_metadatas)
    docsearch.save_local(folder_path=str(faiss_index_path), index_name="Q")
process_and_store_documents([r'C:\Users\ASUS\Desktop\Langchain_Faiss\flask-server\分行邦妮QA_0809_Q.csv'])
# 讀取已經創建的向量數據庫
index = faiss.read_index(str(index_path))
# 獲取向量數據庫中的向量數量
num_vectors = index.ntotal
print("向量數據庫中的向量數量：", num_vectors)

