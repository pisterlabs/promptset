import logging
import pandas as pd
import os
import chromadb
from chromadb.utils import embedding_functions
from openai_completion import OpenAIAssistant
from utils.utils import replace_placeholders_in_dict, str_to_dict
from prompts import (
    SOURCE_DECISION,
    SQL_PROMPT,
    SQL_PROMPT_FOLLOWUP,
    CSV_PROMPT,
    CHROMA_PROMPT,
)
from utils.sql_utils import return_sqlite_results

hf_api_key = os.getenv("HF_API_KEY")


class QuestionHandler:
    def __init__(
        self, question: str, model: str = "gpt-3.5-turbo", csv_path: str = None
    ):
        self.question = question
        self.model = model
        self.assistant = OpenAIAssistant(model=self.model)
        if csv_path:
            self.csv_path = csv_path
        else:
            self.csv_path = "data/trips.csv"
        chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        hf_ef = embedding_functions.HuggingFaceEmbeddingFunction(
            hf_api_key, model_name="BAAI/bge-base-en-v1.5"
        )
        self.collection = chroma_client.get_or_create_collection(
            "country_information", embedding_function=hf_ef
        )

    def get_answer(self):
        """Chain calls to the LLM to get the answer to the question.
        It will chose between 3 sources: CSV, SQL, Chroma (vectorstore)"""
        # Decide source to use
        source_prompt = replace_placeholders_in_dict(
            SOURCE_DECISION, {"question": self.question}
        )
        response = self.assistant.get_openai_completion(**source_prompt)
        logging.debug(f"Full response from {self.model}: {response}")
        logging.debug(f"Number of tokens")
        source_response_content = response["choices"][0]["message"]["content"]
        logging.debug(f"source_reply: {source_response_content}")

        if "csv" in source_response_content.lower():
            logging.debug("CSV source selected")
            _df = pd.read_csv(self.csv_path)
            csv_context = _df.to_records().tolist()
            logging.debug(f"csv_context: {csv_context}")
            csv_prompt = replace_placeholders_in_dict(
                CSV_PROMPT, {"question": self.question, "data": csv_context}
            )

            final_response = self.assistant.get_openai_completion(**csv_prompt)
            logging.debug(f"final_response: {final_response}")
            final_response_content = final_response["choices"][0]["message"]["content"]

        if "sql" in source_response_content.lower():
            logging.debug("SQL source selected")
            sql_prompt = replace_placeholders_in_dict(
                SQL_PROMPT, {"question": self.question}
            )
            sql_response = self.assistant.get_openai_completion(**sql_prompt)
            logging.debug(f"sql_response: {sql_response}")

            # query the database
            results_database = return_sqlite_results(
                query=sql_response["choices"][0]["message"]["content"]
            )
            logging.debug(f"results_database: {results_database}")
            # follow up question

            sql_prompt_followup = replace_placeholders_in_dict(
                SQL_PROMPT_FOLLOWUP,
                {
                    "question": self.question,
                    "query": sql_response["choices"][0]["message"]["content"],
                    "result": results_database,
                },
            )
            sql_response_followup = self.assistant.get_openai_completion(
                **sql_prompt_followup
            )
            logging.debug(f"sql_response_followup: {sql_response_followup}")
            final_response_content = sql_response_followup["choices"][0]["message"][
                "content"
            ]

        if "chroma" in source_response_content.lower():
            # Query the database to find relevant chunks
            logging.debug("Chroma source selected")
            results_collection_query = self.collection.query(
                query_texts=[self.question], n_results=3
            )
            logging.debug(f"results_collection_query: {results_collection_query}")
            results_chroma_documents = results_collection_query["documents"]

            # Call the assistant
            chroma_prompt = replace_placeholders_in_dict(
                CHROMA_PROMPT,
                {"question": self.question, "data": results_chroma_documents},
            )
            logging.debug(f"chroma_prompt: {chroma_prompt}")
            chroma_response = self.assistant.get_openai_completion(**chroma_prompt)
            logging.debug(f"chroma_response: {chroma_response}")
            final_response_content = chroma_response["choices"][0]["message"]["content"]

        else:
            logging.debug("No local source contains the answer, using wikipedia")
            pass

        # parse final response
        final_response_content = str_to_dict(final_response_content)

        return final_response_content
