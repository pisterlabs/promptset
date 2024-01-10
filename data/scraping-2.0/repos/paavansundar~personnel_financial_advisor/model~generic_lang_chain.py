import os
import openai
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
_checkpoint = "gpt2"
class GenericLangChain:

    def loadPDFDB(self):
     vectordb=""
     try:
        # Load PDF
        embedding = OpenAIEmbeddings()
        loaders = [
            # Duplicate documents on purpose - messy data
            PyPDFLoader("./datasets/iinvestor.pdf"),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        # Split
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 20
        )
        splits = text_splitter.split_documents(docs)

        persist_directory = './datasets'

        vectordb = Chroma.from_documents(
            documents=splits, # splits we created earlier
            embedding=embedding,
            persist_directory=persist_directory # save the directory
            )
     except Exception as e:
        print(e)
     return vectordb
    def chat(self,question):
        os.environ['openai_api_key']=""
        embedding = OpenAIEmbeddings()
        vectordb=self.loadPDFDB()
        result=""
        #vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        llm = ChatOpenAI(model_name=_checkpoint, temperature=0)
        if vectordb != None:
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(),return_source_documents=True,
            chain_type_kwargs={"prompt": self.getPromptTemplate()})
            result = qa_chain({"query": question})
        return result["result"]

    def getPromptTemplate(self):
       # Build prompt
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}
            Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        return QA_CHAIN_PROMPT


