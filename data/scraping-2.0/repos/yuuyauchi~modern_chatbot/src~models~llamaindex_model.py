from dataclasses import dataclass

import openai
import pandas as pd
import tinysegmenter
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.evaluation import DatasetGenerator, ResponseEvaluator
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from models.chatbot_base import ChatbotTrainingBase

# from preprocessing.textsplit import TinySegmenterTextSplitter
from utils import setting

segmenter = tinysegmenter.TinySegmenter()

env = setting()
openai.api_key = env["OPENAI_API_KEY"]


@dataclass
class LlamaindexChatBot(ChatbotTrainingBase):
    data_path: str = "sample_data"
    model_name: str = "gpt-3.5-turbo"
    model_path: str = "storage"

    def read_data(self):
        self.documents = SimpleDirectoryReader(input_dir=self.data_path).load_data()

    def preprocess(self):
        self.text_splitter = TokenTextSplitter(
            separator="。",
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            backup_separators=["、", "\n"],
        )

        self.node_parser = SimpleNodeParser(text_splitter=self.text_splitter)
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=self.model_name))
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, node_parser=self.node_parser
        )

    def generate_engine(self):
        self.index = GPTVectorStoreIndex.from_documents(
            self.documents, service_context=self.service_context
        )
        self.query_engine = self.index.as_query_engine(service_context=self.service_context)
        self.index.storage_context.persist(self.model_path)

    def save_question_and_answer(self, file_name: str):
        data_generator = DatasetGenerator.from_documents(self.documents)
        eval_questions = data_generator.generate_questions_from_nodes(50)
        eval_df = pd.DataFrame(
            {
                "query": eval_questions,
            }
        )
        eval_df.to_csv(file_name, index=False)

    def evaluate(self, file_name: str):
        df = pd.read_csv(file_name)
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=self.model_path),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=self.model_path),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=self.model_path),
        )
        vector_store_index = load_index_from_storage(
            storage_context, service_context=self.service_context
        )
        self.query_engine = vector_store_index.as_query_engine(service_context=self.service_context)
        self.evaluator = ResponseEvaluator(service_context=self.service_context)
        df["response"] = df["query"].apply(lambda query: self.query_engine.query(query))
        df["label"] = df["response"].apply(lambda response: str(self.evaluator.evaluate(response)))
        df.to_csv(file_name, index=False)

    @classmethod
    def deploy(cls):
        text_splitter = TokenTextSplitter(
            separator="。",
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            backup_separators=["、", "\n"],
        )

        node_parser = SimpleNodeParser(text_splitter=text_splitter)
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, node_parser=node_parser
        )
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="storage"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir="storage"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="storage"),
        )
        vector_store_index = load_index_from_storage(
            storage_context, service_context=service_context
        )
        query_engine = vector_store_index.as_query_engine(service_context=service_context)
        return query_engine
