import os
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

class Retriever:
    
    def __init__(self, data_source, embedding_model, embedding_model_name, search_kwargs, search_type, query):
        self.data_source = data_source
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.search_kwargs = search_kwargs
        self.search_type = search_type
        self.query = query

    def chroma(self, data_source, embedding, embedding_model_name, search_kwargs, search_type, query):
        
        persist_directory = './backend/db/'+self.data_source+'_chroma_'+self.embedding_model_name

        vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_model)
        retriever=vectordb.as_retriever(search_kwargs=self.search_kwargs, search_type=self.search_type)
        docs = retriever.get_relevant_documents(self.query)

        return docs

    def faiss(self, data_source, embedding, embedding_model_name, search_kwargs, search_type, query):

        docsearch = FAISS.load_local('./backend/db/'+self.data_source+'_faiss_'+self.embedding_model_name, embeddings=self.embedding_model)

        retriever=docsearch.as_retriever(search_kwargs=self.search_kwargs, search_type=self.search_type)
        retriever_2=docsearch.as_retriever(search_kwargs=dict(k=5))

        docs = retriever.get_relevant_documents(self.query)

        # return retriever
        return docs