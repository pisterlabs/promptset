import faiss
import numpy as np
import math, os, random, csv, getpass
from transformers import AutoTokenizer, AutoModel
import torch
#from text2vec import SentenceModel
#from sentence_transformers import SentenceTransformer
import re
from langchain.vectorstores import FAISS
import faiss
import re
import numpy as np
import torch
#from sentence_transformers import SentenceTransformer
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader, TextLoader
from typing import List
from pathlib import Path

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
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
index_path = faiss_index_path / 'index_QA.faiss'
#if not index_path.exists():
 #   texts = ['這是一個測試文本', '這是另一個測試文本']
 #   db = FAISS.from_texts(texts, embeddings)
 #   db.save_local(folder_path=str(faiss_index_path),index_name="index_flowchart")
def process_and_store_documents(file_paths: List[str]) -> None:
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
        loader = CSVLoader(f'{file_pth}', encoding = 'utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=0, separators=["\n", "\n\n", "\t", "。"]
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
    docsearch = FAISS.load_local(str(faiss_index_path), embeddings, index_name="index_QA")
    new_metadatas = [doc.metadata for doc in doc_chunks]
    #管理資料庫
    new_doc_chunk = []
    for i in doc_chunks:
        similarity = docsearch.similarity_search_with_score(i.page_content)[0][1]
        if similarity >= 0.001:
            new_doc_chunk.append(i)
    new_clean_metadatas = [doc.metadata for doc in new_doc_chunk]
    # 提取每個文檔的實際來源
    docsearch.add_texts([t.page_content for t in new_doc_chunk], metadatas=new_clean_metadatas)
    docsearch.save_local(folder_path=str(faiss_index_path), index_name="index_QA")
#process_and_store_documents([r'C:\Users\ASUS\Desktop\Langchain_Faiss\flask-server\flowchart.csv'])
# 讀取已經創建的向量數據庫
index = faiss.read_index(r"C:\Users\ASUS\Desktop\Langchain_Faiss\flask-server\faiss_index\index_QA.faiss")
# 獲取向量數據庫中的向量數量
num_vectors = index.ntotal
print("向量數據庫中的向量數量：", num_vectors)