import datetime

from langchain import PromptTemplate

from config.config import config

basic_prompt = """You are an %s expert. Given an input question, first create a syntactically correct %s query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the TOP clause as per %s. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table. Use Russian language.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here
"""

prompt_suffix = """Only use the following tables:
{table_info}

Question: {input}"""


class DatastepPrompt:
    @classmethod
    def get_prompt(cls, table_description: str = None) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["input", "table_info", "top_k"],
            template=cls.build_prompt(table_description)
        )

    @classmethod
    def build_prompt(cls, table_description) -> str:
        table_description = table_description or config["prompt"]["table_description"]
        table_description = table_description.replace("[[currentDate]]", str(datetime.date.today()))
        db_driver = config["db_driver"]
        return basic_prompt % (db_driver, db_driver, db_driver) + table_description + prompt_suffix
