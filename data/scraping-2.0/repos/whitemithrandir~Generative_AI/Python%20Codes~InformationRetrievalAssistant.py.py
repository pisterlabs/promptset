import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai.api_key  = os.environ['OPENAI_API_KEY']


class Embedding:

    def __init__(self, path):
        self.path = path

    # def process_data(self):
    #     loaders = [
    #         PyPDFLoader(f"{self.path}/docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    #         PyPDFLoader(f"{self.path}/docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    #         PyPDFLoader(f"{self.path}/docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    #         PyPDFLoader(f"{self.path}/docs/cs229_lectures/MachineLearning-Lecture03.pdf")
    #     ]

    #     docs_ = []

    #     for loader in loaders:
    #         docs_.extend(loader.load())

    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1500,
    #         chunk_overlap=150
    #     )

    #     splits = text_splitter.split_documents(docs_)


    def VectorStoreEmbedding(self):

        persist_directory = f'{self.path}docs/chroma/'

        embedding = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        return vectordb, embedding

        # question = "is there an email i can ask for help"
        # result_docs = vectordb.similarity_search(question, k=3)

        # return result_docs

class Retrieval:

    def __init__(self, path):
        self.path = path


        # if os.path.isdir(os.path.abspath(persist_directory)) == True:

    
    def metadata(self):
        x = Embedding(path)
        vectordb, embedding = x.VectorStoreEmbedding() 

        metadata_field_info = [
            AttributeInfo(
                name="source",
                description=f"The lecture the chunk is from, should be one of `{self.path}/docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `{self.path}/docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `{self.path}/docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
                type="string",
            ),
            AttributeInfo(
                name="page",
                description="The page from the lecture",
                type="integer",
            ),
        ]

        document_content_description = "Lecture notes"

        llm = OpenAI(temperature=0)
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vectordb,
            document_content_description,
            metadata_field_info,
            verbose=True
        )

        return retriever

# Kullanım
if __name__ == "__main__":
    path = "/home/sabankara/coding/Learning Platform Python/"
    embedding_instance = Retrieval(path)
    retriever = embedding_instance.metadata()
    question = "what did they say about regression in the third lecture?"
    print(retriever.get_relevant_documents(question))

    c=5
