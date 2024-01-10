import openai
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from CustomRetriever import CustomRetriever

import config


class llm_engine:
    def __init__(self):
        # Initialize the environment variables
        self.setup()
        self.database = self.load_database()
        self.retriever = self.load_retriever(self.database)
        self.llm = self.load_llm()
        self.qa = self.create_search_engine(self.llm, self.retriever)

    def setup(self):
        try:
            load_status = load_dotenv(find_dotenv())
            print(f"load_dotenv: {load_status}\n")
        except Exception as e:
            print(e)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def load_database(self):
        VECTORDB = 'vectordb'
        embeddings = OpenAIEmbeddings()
        vector_store_dir = f"embeddings/{VECTORDB}/"
        vectordb = Chroma(
            persist_directory=vector_store_dir, embedding_function=embeddings
        )

        return vectordb

    def load_retriever(self, database):
        k = config.SEARCH_DOCUMENTS
        retriever = CustomRetriever(
            vectorstore=database,
            retrieve_type=config.RETRIEVE_TYPE,
            search_type=config.SEARCH_TYPE,
            max_elements=k,                 # This is number of documents to return after filtering
            filter=config.FILTER,
            search_kwargs={"k": k * 10},    # This is number of documents to search through
        )

        return retriever

    def load_llm(self):
        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=config.TEMPERATURE)

    def create_search_engine(self, llm, retriever):
        template_string = config.TEMPLATE_STRING

        prompt_template = PromptTemplate(
            template=template_string, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": prompt_template}

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=config.CHAIN_TYPE,
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )
        return qa

    def ask_question(self, question):
        result = self.qa(question)
        return result
