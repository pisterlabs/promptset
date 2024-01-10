import os
import openai
import pinecone
from PyPDF2 import PdfReader

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os



class My_own_gpt():
    def __init__(self, pdf, models, API, api_key, env, index_name):
        self.pdf = pdf
        self.models = models
        self.api_key = api_key
        self.env = env

        self.index_name = index_name
        os.environ["OPENAI_API_KEY"] = API
        self.pdf_text = self.get_pdf_text()
        self.docs = self.split_docs()
        self.embeddings = self.embedding()
        self.index = self.pinecone_vec_DB()




    def split_docs(self, chunk_size=1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_text(self.pdf_text)
        return docs

    def embedding(self):

        embeddings = OpenAIEmbeddings(model=self.models)
        query_result = embeddings.embed_query("Hello world")
        #        print(len(query_result))
        return embeddings

    def pinecone_vec_DB(self):
        pinecone.init(
            api_key=self.api_key,
            environment=self.env
        )
        index = Pinecone.from_texts(self.docs, self.embeddings, index_name=self.index_name)
        return index

    def get_pdf_text(self):
        text = ""
        for Pdf in self.pdf:
            pdf_reader = PdfReader(Pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text