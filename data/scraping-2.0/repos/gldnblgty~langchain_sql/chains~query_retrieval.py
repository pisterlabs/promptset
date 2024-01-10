from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from chains.query_prompt import LITERATURE_PROMPT

class Chain:
    def __init__(self, llm:BaseLanguageModel) -> None:
        self.db = SQLDatabase.from_uri("sqlite:///books.sqlite")
        self.llm = llm
        
    def query(self, query: str) -> str:
        db_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=True)
        
        prompt = PromptTemplate.from_template(LITERATURE_PROMPT).format(input=query)
        return db_chain.run(prompt)