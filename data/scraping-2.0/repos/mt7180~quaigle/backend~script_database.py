# https://python.langchain.com/docs/expression_language/cookbook/sql_db
import logging
import re
import sys
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

from langchain.callbacks import get_openai_callback

from operator import itemgetter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger(__name__).addHandler(logging.StreamHandler(stream=sys.stdout))


class CustomTokenCounter:
    def __init__(self):
        self._total_llm_token_count = 0

    @property
    def total_llm_token_count(self):
        return self._total_llm_token_count

    def reset_counts(self) -> None:
        self._total_llm_token_count = 0

    def add_count(self, value: int) -> None:
        if isinstance(value, int) and value >= 0:
            self._total_llm_token_count += value
        else:
            raise ValueError(
                """Invalid value to add to total_llm_token_count. 
                Count must be a non-negative integer."""
            )


class AIDataBase(SQLDatabase):
    """querying a database with langchain SQLDatabaseChain and Runnables"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = "database"
        self.summary = (
            "Table Info: "
            + re.sub(r"/\*((.|\n)*?)\*/", "", self.get_table_info()).strip()
        )

    def get_schema(self, _):
        return self.get_table_info()

    def run_query(self, working_dict):
        logging.debug(f"Query: {working_dict['query']}")
        return self.run(working_dict["query"])

    def ask_a_question(self, question: str, token_callback: CustomTokenCounter) -> str:
        """with langchain SQLDatabaseChain and Runnables"""
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        # logging.debug(self.get_table_info())
        with get_openai_callback() as callback:
            query_generator = (
                RunnableMap(
                    {
                        "schema": RunnableLambda(self.get_schema),
                        "question": itemgetter("question"),
                    }
                )
                | ChatPromptTemplate.from_template(
                    """Based on the table schema below, write a SQL query that 
                    would answer the user's question:
                    {schema}

                    Question: {question}
                    SQL Query:"""
                )
                | llm.bind(stop=["\nSQLResult:"])
                | StrOutputParser()
                | {"query": RunnablePassthrough()}
            )

            chain = (
                query_generator
                | {
                    "response": RunnableLambda(self.run_query),
                    "question": RunnablePassthrough(),
                    "query": RunnablePassthrough(),
                }
                | ChatPromptTemplate.from_template(
                    """Based on the question and the sql response, 
                    write a natural language response and finally add 
                    the sql query to your response:

                    Question: {question}
                    SQL Response: {response}
                    Query: {query}"""
                )
                | llm
            )
            response = chain.invoke({"question": question})
            token_callback.add_count(callback.total_tokens)
        return response.content


class DataChatBotWrapper:
    def __init__(self, callback_manager: CustomTokenCounter):
        self.data_category: str = "database"
        self.token_callback = callback_manager
        self.document: AIDataBase = None

    def add_document(self, document) -> None:
        self.document = document

    def clear_chat_history(self) -> str:
        return "No chat history available for database"

    def clear_data_storage(self) -> None:
        del self.document
        self.document = None
        # ToDo delete db file ?

    def update_temp(self, temperature) -> None:
        pass

    def answer_question(self, question: str) -> str:
        return self.document.ask_a_question(question, self.token_callback)


def set_up_database_chatbot():
    token_counter = CustomTokenCounter()
    return (
        DataChatBotWrapper(callback_manager=token_counter),
        None,
        token_counter,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    import certifi
    import os

    # load your API key to the environment variables
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    # workaround for mac to solve SSL: CERTIFICATE_VERIFY_FAILED Error
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()

    chat_engine, callback_manager, token_counter = set_up_database_chatbot()

    document: AIDataBase = AIDataBase.from_uri("sqlite:///data/database.sqlite")
    print("________")
    print(document.summary)

    chat_engine.add_document(document)

    question = (
        """Which name has the user who wrote the largest amount of helpful reviews?"""
    )
    print(document.summary)
    print(chat_engine.answer_question(question))
    logging.debug(f"Number of used tokens: {token_counter.total_llm_token_count}")
    # print(document.get_table_info())
