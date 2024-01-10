from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# 创建语言模型
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 创建文本加载器
loader = UnstructuredEPubLoader('../elon_mask.epub')
documents = loader.load()
# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# 将文本分割成小块
texts = text_splitter.split_documents(documents)
# 创建向量存储，整理流程：1. 将文档分成小块 2. 将小块文档创建Embedding 3. 将向量存储
db = Chroma.from_documents(texts, embeddings,persist_directory='db')
db.persist()
db = None