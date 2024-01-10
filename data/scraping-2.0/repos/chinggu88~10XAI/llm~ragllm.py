
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from llm.llmcallback import ChatCallbackHandler
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

import streamlit as st
import pymysql
import pandas as pd

class ragllm:
    def __init__(self):
        OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
        HOST=st.secrets["host"]
        PORT=st.secrets["port"]
        USER=st.secrets["user"]
        PASSWORD=st.secrets["password"]
        DATABASE=st.secrets["database"]
        MYSQL_URL=st.secrets["MYSQL_URL"]
       
        self.connection = pymysql.connect(
            host=HOST,
            port=int(PORT),
            user=USER,
            password =PASSWORD,
            database=DATABASE
        )
        self.llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo',verbose=True,streaming=True,)

        self.llmex = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo',verbose=True,streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],)

        self.db = SQLDatabase.from_uri(MYSQL_URL)
        self.db_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=True)
        
    @st.cache_resource(show_spinner="Embedding file...")
    def embed_file(file):
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
    
    def run_query(self,query):
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute(query)
        datas = cursor.fetchall() 
        cursor.close()
        return pd.DataFrame(datas)
    
    def dbinfo(self):
        return self.db.table_info