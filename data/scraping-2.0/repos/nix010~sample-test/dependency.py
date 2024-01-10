import os

import openai
from llama_index import SQLDatabase
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_query_engine():
    sql_database = SQLDatabase(engine, include_tables=["clubs"])
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["clubs"],
    )
    return query_engine
