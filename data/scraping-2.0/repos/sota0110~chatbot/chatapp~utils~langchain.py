import sys
sys.path.append('../../')
import os
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from ..config.openai_key import free_secret_key
from chatbot.settings import DATABASES

class LangChain:
    def __init__(self) -> None:
        os.environ["OPENAI_API_KEY"] = free_secret_key
        llm = OpenAI(model_name = "text-davinci-002", temperature = 0, verbose = False)
        db_user = DATABASES["default"]["USER"]
        db_password = DATABASES["default"]["PASSWORD"]
        db_host = DATABASES["default"]["HOST"]
        db_name = DATABASES["default"]["NAME"]
        database = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        
        template = \
            """
            Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Use the following format:
                Question:   "Question here"
                SQLQuery:   "SQL Query to run"
                SQLResult:  "Result of the SQLQuery"
                Answer:     "Final answer here"
            Only use the following tables: {table_info}
            Question: {input}
            """
        
        prompt = PromptTemplate(input_variables = ["input", "table_info", "dialect"], template = template)
        
        self.db_chain = SQLDatabaseChain.from_llm(llm, database, prompt=prompt, verbose=True)
    
    def search_db(self, text):
        result = self.db_chain.run(text)
        
        return result