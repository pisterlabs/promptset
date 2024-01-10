from databricks import sql
import os
import json
import re

from app.configurations.development.config_parser import args
from app.engine.query_generation_engine import QueryGenerationEngine
from app.engine.memory_engine import MemoryEngine
from app.engine.prompts_engine import PromptsEngine
from app.engine.postprocessing_engine import PostProcessingEngine
from app.engine.sessions_manager_engine import SessionsManagerEngine
from app.engine.document_retrieval_engine import DocumentRetrievalEngine
from app.configurations.development.settings import (
    FALLBACK_MESSAGE,
    SERVER_OVERLOAD_MESSAGE,
    TOP_K_EMBEDS,
    TEMPERATURE,
)

from pydantic import Field
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import BaseChatMessageHistory
from utils.callbacks.logging_callback_handler import LoggingCallbackHandler
from utils.utils import get_embeds_with_scores, get_columns_from_docs


class AskSkanQAEmbedManager:
    def __init__(
        self,
        memory_engine: MemoryEngine,
        document_retrieval_engine: DocumentRetrievalEngine,
        prompts_engine: PromptsEngine,
        query_generation_engine: QueryGenerationEngine,
        post_processing_engine: PostProcessingEngine,
        logging_callback_handler: LoggingCallbackHandler = LoggingCallbackHandler(),
    ):
        self.memory_engine = memory_engine
        self.document_retrieval_engine = document_retrieval_engine
        self.prompts_engine = prompts_engine
        self.query_generation_engine = query_generation_engine
        self.postprocessing_engine = post_processing_engine
        self.logging_callback_handler = logging_callback_handler

        self.retriever = None

    def _execute_pyspark_query(self, query, logger):
        DBWS_HOST = os.getenv("dbws_host_domain")
        DBWS_HTTP_PATH = os.getenv("dbws_host_path")
        DBWS_PAT = os.getenv("dbws_pat")

        result_rows = []

        with sql.connect(
            server_hostname=DBWS_HOST, http_path=DBWS_HTTP_PATH, access_token=DBWS_PAT
        ) as conn:
            with conn.cursor() as cursor:
                logger.logger.debug("Executing query on the datalake server...")
                cursor.execute(query)
                result = cursor.fetchall()
                logger.logger.debug("Query executed successfully and results fetched!")

                for row in result:
                    result_rows.append(row)
        if len(result_rows) == 1:
            return result_rows[0]

        return result_rows

    def get_qa_chain(
        self,
        model_verbose=False,
    ):
        # vectorstore = self.document_retrieval_engine.get_vectorstore_data()
        self.retriever = (
            self.document_retrieval_engine.get_vectorstore_data().as_retriever()
        )
        combine_docs_chain_kwargs = self.query_generation_engine.get_chain_type_kwargs(
            instruction_prompt=self.prompts_engine.get_qa_prompt()
        )

        question_gen_llm = self.query_generation_engine.get_question_generation_llm()
        combine_docs_llm = self.query_generation_engine.get_combine_doc_llm(
            streaming=True, temperature=args.temperature
        )
        buffer_memory = self.memory_engine.get_buffer_memory()

        qa_chain = self.query_generation_engine.get_query_generation_chain(
            memory=buffer_memory,
            combine_doc_llm=combine_docs_llm,
            retriever=self.retriever,
            question_generation_llm=question_gen_llm,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs,
            verbose=model_verbose,
            tracing=False,
            # callbacks=[self.logging_callback_handler],
        )

        return qa_chain, buffer_memory

    def get_qa_embed_docs(self, user_input, qa_chain):
        generated_query = None
        try:
            # Entering QA retrieval chain...
            generated_response = qa_chain({"question": user_input})
            generated_output = generated_response["answer"].strip()

        except Exception:
            # Failed to retrieve model response
            return "N/A", "N/A", "N/A"

        # Fetching relevant columns from schema...
        docs_and_similarities = (
            self.retriever.vectorstore.similarity_search_with_relevance_scores(
                user_input, k=TOP_K_EMBEDS
            )
        )
        docs, embed_ranks, sim_scores, embed_rank_sim = get_embeds_with_scores(
            docs_and_similarities
        )
        doc_cols = get_columns_from_docs(docs)

        try:
            # Extracting SQL query from JSON reponse..."
            generated_query, is_query = self.postprocessing_engine.extract_sql_query(
                generated_output
            )

            if not is_query:
                return generated_query, "N/A", "N/A"
                # Not a SQL query, returning statement as is

        except Exception as e:
            return FALLBACK_MESSAGE, "N/A", "N/A"

        return (
            generated_query,
            doc_cols,
            embed_rank_sim,
        )
