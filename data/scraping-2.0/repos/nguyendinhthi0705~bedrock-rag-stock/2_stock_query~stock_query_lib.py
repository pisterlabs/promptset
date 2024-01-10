import os
from langchain.llms.bedrock import Bedrock
import json
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime, timedelta
from datetime import date

def get_llm():
        
    model_parameter = {"temperature": 0.0, "top_p": .5, "max_tokens_to_sample": 2000}
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), 
        region_name=os.environ.get("BWB_REGION_NAME"), 
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), 
        model_id="anthropic.claude-v2", 
        model_kwargs=model_parameter) 
    
    return llm


def get_db_chain(prompt):
    db = SQLDatabase.from_uri("sqlite:///stock_ticker_database.db")
    llm = get_llm()
    db_chain = SQLDatabaseChain.from_llm(
        llm, 
        db, 
        verbose=True, 
        return_intermediate_steps=True, 
        prompt=prompt,
        return_direct=True,
    )
    return db_chain
    
    
def query_stock(query):
   
    llm = get_llm()
    _DEFAULT_TEMPLATE = """Human: Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
<format>
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "SQLQuery and Result of SQLResult"
</format>
Assistant: Understood, I will use the above format and only provide the answer.

Only use the following tables:
<tables>
CREATE TABLE stock_ticker (
	symbol text PRIMARY KEY,
	name text NOT NULL,
	currency text,
	stockExchange text, 
    exchangeShortName text
)
</tables>

If someone asks for the table stock ticker table, they really mean the stock_ticker table.
<examples>
Question: 
        What is the ticker symbol for Amazon?
        Params: 
        Company name (name): Amazon
        
SQLQuery:SELECT * FROM stock_ticker WHERE name LIKE '%Amazon%'

</examples>

Question: \n\nHuman:{input} \n\nAssistant:

"""

    PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
)
    db_chain = get_db_chain(PROMPT)
    return db_chain("\n\nHuman: What is the ticker symbol for " + str(query) + " ? \n\nAssistant:")
