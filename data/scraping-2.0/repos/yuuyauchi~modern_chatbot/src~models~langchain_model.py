# import os
from dataclasses import dataclass

import openai

# import pandas as pd
import tinysegmenter

# from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

# from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from models.chatbot_base import ChatbotTrainingBase
from utils import setting

segmenter = tinysegmenter.TinySegmenter()

env = setting()
openai.api_key = env["OPENAI_API_KEY"]


@dataclass
class LangChainChatBot(ChatbotTrainingBase):
    data_path: str = "sample_data"
    model_name: str = "gpt-3.5-turbo"

    def read_data(self):
        self.loader = DirectoryLoader(self.data_path)

    def preprocess(self):
        # TODO 正答の精度を上げるために異なるsplitterも試す。
        self.text_splitter = CharacterTextSplitter(
            separator="。",
            chunk_size=1000,
            chunk_overlap=50,
        )

    def generate_engine(self):
        self.index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=OpenAIEmbeddings(),
            text_splitter=self.text_splitter,
        ).from_loaders([self.loader])
        return self.index
        # self.index.query()
        # self.chain = RetrievalQA.from_chain_type(
        #     llm=OpenAI(),
        #     chain_type="stuff",
        #     retriever=self.index.vectorstore.as_retriever(),
        #     input_key="question",
        #     return_source_documents=True
        # )

    def evaluate(self):
        pass
