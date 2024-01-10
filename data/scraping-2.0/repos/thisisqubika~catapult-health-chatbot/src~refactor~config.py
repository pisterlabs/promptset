# config.py
from dotenv import load_dotenv, find_dotenv
from langchain.sql_database import SQLDatabase
import os
import streamlit as st
from snowflake.snowpark import Session
from sqlalchemy.dialects import registry
from snowflake.connector import connect, ProgrammingError


load_dotenv(find_dotenv())
registry.load('snowflake')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACCOUNT = os.environ.get("ACCOUNT")
USER = os.environ.get("USER")
PASSWORD = os.environ.get("PASSWORD")
SCHEMA = os.environ.get("SCHEMA")
WAREHOUSE = os.environ.get("WAREHOUSE")
ROLE = os.environ.get("ROLE")
DATABASE = os.environ.get("DATABASE")


# create connection:
snowflake_url = f"snowflake://{USER}:{PASSWORD}@{ACCOUNT}/{DATABASE}/{SCHEMA}?warehouse={WAREHOUSE}&role={ROLE}"
db = SQLDatabase.from_uri(snowflake_url)


global snowflake_session
snowflake_session = None

def create_session():
    try:
        snowflake_session = Session.builder.configs({
            "account": ACCOUNT,
            "user": USER,
            "password": PASSWORD,
            "role": ROLE,
            "warehouse": WAREHOUSE,
            "database": DATABASE,
            "schema": SCHEMA
        }).create()

    except ProgrammingError as e:
        if "Authentication token has expired" in str(e):
            snowflake_session = Session.builder.configs({
                "account": ACCOUNT,
                "user": USER,
                "password": PASSWORD,
                "role": ROLE,
                "warehouse": WAREHOUSE,
                "database": DATABASE,
                "schema": SCHEMA
            }).create()
        
        else:
            raise  # Re-raise the exception if it's not a token expiration issue

    return snowflake_session

