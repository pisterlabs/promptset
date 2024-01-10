from email import generator
import os
from langchain import VectorDBQA
from llama_index import OpenAIEmbedding
from langchain.vectorstores import Chroma

from config import Config
from services.ThreadedGenerator import ChainStreamHandler
from vectordb.BaseVector import BaseVector
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain import OpenAI, VectorDBQA

from langchain.embeddings.openai import OpenAIEmbeddings


class ChromaDb(BaseVector):

    _clientMap = dict()

    def get_client(self, index: str, doc_path: str=None, reload: bool=False):
        print('get_client----',index)
        print('_clientMap----',self._clientMap)
        if index in self._clientMap and reload is False:
            return self._clientMap[index]
        
        root_path = self.get_rootpath(index)

        if doc_path is not None and reload:
            if os.path.exists(doc_path):
                client = self.create_index_text(index, doc_path)
                self._clientMap[index] = client
                return client

        embeddings = self.get_embedding()
        client = Chroma(persist_directory=root_path, embedding_function=embeddings)
        self._clientMap[index] = client
        return client
    
    def get_rootpath(self, index: str):
        return Config.VECTOR_STORE_PATH+"/"+index

    def create_index_text(self, index, doc_path):
        root_path = self.get_rootpath(index)

        if os.path.exists(root_path):
            self.clearFiles(root_path)
        print('create_index_chromadb')
         # 加载文件夹中的所有txt类型的文件
        split_docs = self.get_docs(doc_path)
        print("split_docs======",split_docs)

        # 初始化 openai 的 embeddings 对象
        embeddings = self.get_embedding()
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        client = Chroma.from_texts([t.page_content for t in split_docs], embeddings, persist_directory=root_path)
        client.persist()
        return client
    
    def create_index_documents(self, index,doc_path,client=None):
        root_path = self.get_rootpath(index)
        if os.path.exists(root_path):
            return True
        print('create_index_chromadb')
         # 加载文件夹中的所有txt类型的文件
        split_docs = self.get_docs(doc_path)
        print("split_docs----",split_docs)

        # 初始化 openai 的 embeddings 对象
        embeddings = self.get_embedding()
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        client = Chroma.from_documents([t.page_content for t in split_docs], embeddings, persist_directory=root_path)
        client.persist()
        return True
    
    @classmethod
    def get_embedding(self):
        return OpenAIEmbeddings()
    
    def retriever_search(self, query: str, k: int = 1, client=None, search_type="mmr"):
        retriever = client.as_retriever(search_type=search_type,search_kwargs={"k":k})# search_type="mmr", k=10, alpha=0.5, beta=0.5)
        docs = retriever.get_relevant_documents(query)
        return docs
    
    def qa(self, query: str, k: int = 1,client=None):
        llm = ChatOpenAI(temperature=0, verbose=True)
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=client, return_source_documents=True)
        return qa({"query": query})
    
    def qaStream(self, query: str, k: int = 1,client=None):
        llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager([ChainStreamHandler(generator)]), verbose=True)
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=client, return_source_documents=True)
        return qa({"query": query})
    
    def getRetrievalQA(self,client):
       return RetrievalQA.from_chain_type(llm=OpenAI(verbose=True), chain_type="stuff", retriever=client.as_retriever())

    def getRetrievalQAStream(self, client):
       llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager([ChainStreamHandler(generator)]), verbose=True)
       return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=client.as_retriever())

    #return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    #删除文件夹下所有文件
    def clearFiles(self, rootdir):
        print('clearFiles----',rootdir)
        filelist = os.listdir(rootdir)
        for f in filelist:
            filepath = os.path.join(rootdir, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                self.clearFiles(filepath)
                os.rmdir(filepath)
        return True

                    