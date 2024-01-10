from datetime import datetime

from sqlalchemy.engine import create_engine
from trino.sqlalchemy import URL

from ubix.chain.sql.sql_base import SQLDatabaseChainEx, SQLDatabaseEx
from ubix.common.llm import llm, get_llm
from ubix.common.log_basic import logging

try:
    from langchain import SQLDatabaseChain
except Exception as e:
    from langchain_experimental.sql import SQLDatabaseChain



def get_db_chain(llm):
    hive_host = "trino"
    port = 8080
    user_name = "hive"
    catalog="hive"
    include_tables = ["salesopportunities"]
    hive_database = "65057500bed4c2ac869fe964"
    engine = create_engine(
        URL(
            host=hive_host,
            port=port,
            user=user_name,
            catalog=catalog,
            schema=hive_database,
        ),
    )

    sql_db = SQLDatabaseEx(engine, schema=hive_database, include_tables=include_tables)

    llm = get_llm()

    db_chain = SQLDatabaseChainEx.from_llm(llm=llm, db=sql_db, verbose=True)
    return db_chain



if __name__ == "__main__":

    print(datetime.now())
    start = datetime.now()
    agent = get_db_chain(llm)
    """
    query = "how many records are there in this table?"
    print(datetime.now())
    
    agent.run(query)
    """
    query = "what is the maximum total in this table?"
    answer = agent.run(query)
    print(datetime.now())
    """
    query = "What is the maximum total  in the city Novi"
    agent.run(query)
    print(datetime.now())
    """
    end = datetime.now()
    duration = (end-start).total_seconds()
    logging.info("🔴 answer:\n" + answer)
    logging.info("⏰ " + f"Query: cost:{duration:.0f} sec seconds")



"""
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.  LLM_TYPE=vllm python ubix/chain/chain_sql_ubix.py

PYTHONPATH=.  LLM_TYPE=din python ubix/chain/chain_sql_ubix.py
PYTHONPATH=.  LLM_TYPE=tgi python ubix/chain/chain_sql_ubix.py
"""
