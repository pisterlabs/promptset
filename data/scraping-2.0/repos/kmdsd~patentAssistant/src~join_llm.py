import os
import configparser

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

import chromadb
from langchain.vectorstores import Chroma

# ====================================================================
# 環境変数の取得
# ==================================================================== 
# OpenAI関連
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")

# ====================================================================
# クラス定義
# ==================================================================== 
class join_llm:
    
    def __init__(self, chunkSize, persist_dir):
        
        self.api_type = 'azure'
        self.api_version = '2023-05-15'
        self.api_key = AZURE_OPENAI_KEY
        self.api_base = AZURE_OPENAI_RESOURCE
        self.embeding_deployment = 'text-embedding-ada-002'
        self.embedding_model = 'text-embedding-ada-002'
        self.llm_model = AZURE_OPENAI_MODEL
        self.llm_deployment_id = AZURE_OPENAI_MODEL
        
        os.environ["OPENAI_API_TYPE"] = self.api_type
        os.environ["OPENAI_API_VERSION"] = self.api_version
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_API_BASE"] = f'https://{self.api_base}.openai.azure.com/'
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.chunkSize = chunkSize
        
        # ベクトル化用モデルを定義
        self.embeddings = OpenAIEmbeddings(
            deployment=self.embeding_deployment,
            model=self.embedding_model,
            chunk_size = chunkSize,
        )
        
    # DBの読み込み
    def read_db(self):
        self.db = Chroma(
            collection_name="langchain_store",
            embedding_function=self.embeddings,
            client=self.client,
        )
    
    def join_llm_vector(self):
        print(self.llm_model, self.llm_deployment_id)
        print(len(self.db.get()['ids']))
        llm = ChatOpenAI(model_name=self.llm_model,model_kwargs={"deployment_id":self.llm_deployment_id})
        self.crc = ConversationalRetrievalChain.from_llm(llm,
                                                    self.db.as_retriever(
                                                        search_kwargs={'k':1}
                                                        ),
                                                    return_source_documents=True)
        
    def question(self, query):

        chat_history = []
        result = self.crc({"question": query, "chat_history": chat_history})

        return result["answer"]