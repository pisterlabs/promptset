from datetime import datetime
from functools import lru_cache
from typing import Any 
import re

import langchain
import torch
from langchain.cache import InMemoryCache
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import LLMResult
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain import PromptTemplate, LLMChain
from pydantic import Extra
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain import LlamaCpp, OpenAI
from trino.sqlalchemy import URL

from ubix.common.llm import get_llm

try:
    from langchain import SQLDatabaseChain
except Exception as e:
    from langchain_experimental.sql import SQLDatabaseChain


class SQLEnhancementHandler(BaseCallbackHandler):
    def on_llm_end(
            self,
            response: LLMResult,
            **kwargs: Any,
    ) -> Any:
        sql_cmd = response.generations[0][0].text
        response.generations[0][0].text = re.sub(r';$', '', sql_cmd)


def create_table(table_columns, include_tables, query):
    """
        create a custom table to reduce the reduclant column
        table_columns: origninal table columns
        include_tables: table name
        query: use query to choose the most relevant column

    """
    table_name = include_tables[0]
    custom_table_info = {}
    prefix = "CREATE TABLE " + table_name + " ("
    query_list = query.split(" ")

    mid = ""
    for item in table_columns:
        # mid = ",".join([item for item in query_list if item in query_list])
        if item in query_list:
            mid += item + ","

    last = ")"
    custom_table_info[table_name] = prefix + mid + last
    return custom_table_info

def check_query_or_other(query):
    prompt = PromptTemplate(
    input_variables=["input"],
    template="""
                You are currently doing a classification task, for question about\
                data or table, classify them into Category '''query'''. For other type of questions, \
                classify them into Category '''other'''. Your answer must be only one word, \
                
                Here are a few of examples: \
                
                User: How many records in the table? \
                Assistant: query \
                
                User: What's the max number in table \
                Assistant: query \
                
                User: What's the sells amount in this month. \
                Assistant: query \
                
                User: What's  the average product amount in last year. \
                Assistant: query \
                
                User: who are you? \
                Assistant: other \
                
                User: what is your name? \
                Assistant: other \
                
                User:{input}
                Assistant: """
                )
    search_chain = LLMChain(llm=llm, prompt=prompt)
    label = search_chain.run(query)
    result = label.split("\n")[0]
    return result
    
def get_db_chain(llm, query, database):
    import langchain as lc

    include_tables = [database]
    hive_host = "trino"
    port = 8080
    user_name = "hive"
    catalog="hive"
    hive_database = "63bd509c8cb02db7e453ad27"

    engine = create_engine(
                        URL(
                             host=hive_host,
                             port=port,
                             user=user_name,
                             catalog=catalog,
                             schema=hive_database,
                            ),
                        )
    # query = "what is the maximum total in this table?"
    connetion = engine.connect()

    metadata = MetaData()

    table = Table(include_tables[0], metadata, autoload=True, autoload_with=engine)
    table_columns = table.columns.keys()

    custom_table_info = create_table(table_columns, include_tables, query)
    con_custom = lc.SQLDatabase(engine, include_tables=include_tables, custom_table_info=custom_table_info)
    llm.callbacks=[SQLEnhancementHandler()]
    db_chain = SQLDatabaseChain.from_llm(llm=llm, db=con_custom, verbose=True)
    return db_chain


if __name__ == "__main__":
    llm = get_llm()

    config = {
        "total": "sales_order_item",
    }
    question_round1 = "Hello, I'm Felix"
    question_round2 = "what is the maximum total in this table?"
    question_round3 = "what is the maximum price_total in last year?"
    question_round_list = [question_round1, question_round2, question_round3]
    memory=ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory
    )
    for question_round in question_round_list:
        flag = check_query_or_other(question_round)
        if "query" in flag:
            no_database = False
            database = ""
            for item in question_round.split(" "):
                if item in config:
                    no_database = True
                    database = config[item]
            if not no_database:
                print("choose suitable table")
            else:
                agent = get_db_chain(llm, question_round, database)
                round = agent.run(question_round)
        else:
            round = conversation.predict(input=question_round)
            print("round", round)

"""
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python ubix/chain/chain_sql_ubix.py
"""
