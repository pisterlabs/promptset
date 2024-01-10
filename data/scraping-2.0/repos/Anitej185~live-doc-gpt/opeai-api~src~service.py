import datetime
import os

from langchain.chat_models import ChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import JSONLoader


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import pickle
OPENAI_API_KEY = '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'
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"

class LangChainResponseGenerator():
    def __init__(self, srcs) -> None:
        self.srcs = srcs
        self.embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        cwd = os.path.dirname(os.path.abspath(__file__))
        pwd = os.path.dirname(cwd)
        self.persist_dir = os.path.join(pwd, "persist")

    def preproc(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 150
        )

        pypdfs = []
        pyjson = []
        for src in self.srcs:
            if not os.path.exists(src):
                raise FileNotFoundError(f"{src} file not found!")
            if src.endswith('.pdf'):
                pypdfs.extend(PyPDFLoader(src).load())
            elif src.endswith('.json'):
                pyjson.extend(JSONLoader(src).load())
                pass
            else:
                raise ValueError(f"Unsupported file type: {src}")
            #pypdfs.extend(PyPDFLoader(src).load())
            #pyjson.extend(JSONLoader(src).load())
        
        splits = text_splitter.split_documents(pypdfs)
        return splits

    def run_query(self, sentence):
        return self.embedding.embed_query(sentence)
    
    def create_vector_db(self):
        splits = self.preproc()
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding,
            persist_directory=self.persist_dir
        )
        return vectordb
    
    def create_llm(self):
        llm = ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=OPENAI_API_KEY)
        return llm 
    
# if __name__ == "__main__":

    # srcs = [
    #     "/Users/rohandeshpande/live-doc-gpt/documents/codes.json"
    # ]
    # lcrg = LangChainResponseGenerator(srcs)

    # llm = lcrg.create_llm()
    # vdb = lcrg.create_vector_db()
    
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vdb.as_retriever()
    # )
    
    # user_question = "What is a migraine?"
    # result = qa_chain({"query": user_question})
    
    # print(result["result"])


