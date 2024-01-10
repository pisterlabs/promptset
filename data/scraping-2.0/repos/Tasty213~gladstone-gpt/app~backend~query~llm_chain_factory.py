from pathlib import Path
from fastapi import WebSocket
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from messageData import MessageData
from schema.message import Message
from query.callbacks.final_answer import FinalAnswerCallback
from settings.chat_bot_settings import ChatbotSettings
import os
import boto3

from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from opentelemetry import trace
from query.callbacks.otel_callback import OpentelemetryCallback
from query.callbacks.streaming_callback import StreamingCallback


tracer = trace.get_tracer("chatbot.vortex_query")


class LLMChainFactory:
    USER_PROMPT = "Question:```{question}```"

    def __init__(
        self,
        message_data_table: MessageData,
        vector_store: VectorStore,
        settings: ChatbotSettings,
    ):
        self.message_data_table = message_data_table
        self.vector_store = vector_store
        self.settings = settings

    @staticmethod
    def download_document_store(settings: ChatbotSettings):
        if not Path(settings.persist_directory).exists():
            LLMChainFactory.download_data(settings)

    @staticmethod
    @tracer.start_as_current_span("chatbot.VortexQuery.download_data")
    def download_data(settings: ChatbotSettings):
        """
        Download the contents of a folder directory
        Args:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(settings.document_store_bucket)
        for obj in bucket.objects.filter():
            target = (
                obj.key
                if settings.persist_directory is None
                else os.path.join(settings.persist_directory, obj.key)
            )
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == "/":
                continue
            bucket.download_file(obj.key, target)

    @staticmethod
    @tracer.start_as_current_span("chatbot.VortexQuery.get_vector_store")
    def get_vector_store(settings: ChatbotSettings) -> VectorStore:
        LLMChainFactory.download_document_store(settings)
        embedding = OpenAIEmbeddings(client=None)

        return Chroma(
            collection_name=settings.collection_name,
            embedding_function=embedding,
            persist_directory=settings.persist_directory,
        )

    @staticmethod
    def get_chat_prompt_template(settings: ChatbotSettings) -> ChatPromptTemplate:
        system = settings.system_prompt
        user = LLMChainFactory.USER_PROMPT
        messages = [
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template(user),
        ]
        return ChatPromptTemplate.from_messages(messages)

    @tracer.start_as_current_span("chatbot.VortexQuery.make_chain")
    def make_chain(
        self,
        websocket: WebSocket,
        previous_message: Message,
    ) -> ConversationalRetrievalChain:
        stream_handler = StreamingCallback(websocket)
        final_answer_handler = FinalAnswerCallback(
            websocket, previous_message, self.message_data_table
        )
        otel_handler = OpentelemetryCallback()

        question_gen_llm = OpenAI(
            temperature=self.settings.temperature,
            verbose=True,
            callbacks=[otel_handler],
        )
        question_generator = LLMChain(
            llm=question_gen_llm,
            prompt=CONDENSE_QUESTION_PROMPT,
            callbacks=[otel_handler],
        )

        streaming_llm = ChatOpenAI(
            streaming=True,
            callbacks=[stream_handler, otel_handler],
            verbose=True,
            temperature=self.settings.temperature,
            model=self.settings.model_name,
            max_tokens=self.settings.max_tokens,
        )
        doc_chain = load_qa_chain(
            streaming_llm,
            chain_type="stuff",
            prompt=LLMChainFactory.get_chat_prompt_template(self.settings),
            callbacks=[otel_handler],
        )

        qa = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.settings.documents_returned,
                    "fetch_k": self.settings.documents_considered,
                    "lambda_mult": self.settings.lambda_mult,
                },
            ),
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            return_source_documents=True,
            callbacks=[final_answer_handler, otel_handler],
        )

        return qa
