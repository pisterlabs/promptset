import streamlit as st
import pandas as pd
import logging, sys, os, utils as u

# workaround for https://github.com/snowflakedb/snowflake-sqlalchemy/issues/380.
try:
    u.snowflake_sqlalchemy_20_monkey_patches()
except Exception as e:
    raise ValueError("Please run `pip install snowflake-sqlalchemy`")

import openai
from dotenv import load_dotenv
from llama_index import SQLDatabase, LLMPredictor, SimpleDirectoryReader, VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from sqlalchemy import create_engine
from langchain.chat_models import ChatOpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.indices.loading import load_index_from_storage
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# file path for storing the variables for pfizer_index_id and merck_index_id
variables_file = "variables.txt"

def load_variables():
    global pfizer_index_id, merck_index_id

    # check if the variables file exists
    if os.path.isfile(variables_file):
        with open(variables_file, "r") as file:
            # read the values from the file
            values = file.read().split(",")
            pfizer_index_id = values[0]
            merck_index_id = values[1]

def save_variables():
    global pfizer_index_id, merck_index_id

    # write the values to the file
    with open(variables_file, "w") as file:
        file.write(f"{pfizer_index_id},{merck_index_id}")


def structured_data_querying(structured_question):

    # for connect to Snowflake
    snowflake_uri = "snowflake://<username>:<password>@<org-account>/<database_name>/<schema_name>?warehouse=<warehouse_name>&role=<role_name>"
    
    #define node parser and LLM
    chunk_size = 1024
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm_predictor=llm_predictor)
    
    engine = create_engine(snowflake_uri)

    sql_database = SQLDatabase(engine)

    TEXT_TO_SQL_TMPL = (
        "Given an input question, first create a syntactically correct {dialect} "
        "query to run, then look at the results of the query and return the answer. "
        "You can order the results by a relevant column to return the most "
        "interesting examples in the database.\n"
        "Never query for all the columns from a specific table, only ask for a "
        "few relevant columns given the question.\n"
        "Pay attention to use only the column names that you can see in the schema "
        "description. "
        "Be careful to not query for columns that do not exist. "
        "Pay attention to which column is in which table. "
        "Also, qualify column names with the table name when needed.\n"
        "Use the following format:\n"
        "Question: Question here\n"
        "SQLQuery: SQL Query to run\n"
        "SQLResult: Result of the SQLQuery\n"
        "Answer: Final answer here\n"
        "Only use the tables listed below.\n"
        "{schema}\n"
        "Question: {query_str} \n"
        "Order by revenue value converted to number from highest to lowest. "
        "Please use 'company_name' column from sec_cik_index, use sec_cik_index's SIC_CODE_CATEGORY "
        "of 'Office of Life Sciences' to identify life science companies, use sec_report_attributes's "
        "tag 'Revenues', statement 'Income Statement', metadata IS NULL, value is not null, "
        "period_start_date '2022–01–01' and period_end_date '2022–12–31'. \n"
        "SQLQuery: "
    )

    TEXT_TO_SQL_PROMPT = Prompt(
        TEXT_TO_SQL_TMPL,
        prompt_type=PromptType.TEXT_TO_SQL,
    )

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["sec_cik_index", "sec_report_attributes"],
        service_context=service_context,
        text_to_sql_prompt = TEXT_TO_SQL_PROMPT
    )

    response = query_engine.query(structured_question)
    sql_query = response.metadata["sql_query"]
    print(">>> sql_query: " + sql_query)

    con = engine.connect()
    df = pd.read_sql(sql_query, con)
    st.write(df)
    st.area_chart(df, x="company_name", y="revenue")
    return df


def unstructured_data_querying(unstructured_question):

    # declare the variables as global
    global pfizer_index_id, merck_index_id  
    
    chunk_size = 1024
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm_predictor=llm_predictor)

    try:
        # retrieve existing storage context and load pfizer_index and merck_index
        pfizer_storage_context = StorageContext.from_defaults(persist_dir="./storage_pfizer")
        pfizer_index = load_index_from_storage(storage_context=pfizer_storage_context, index_id=pfizer_index_id)
        
        merck_storage_context = StorageContext.from_defaults(persist_dir="./storage_merck")
        merck_index = load_index_from_storage(storage_context=merck_storage_context, index_id=merck_index_id)
        logging.info("pfizer_index and merck_index loaded")
        
    except FileNotFoundError:
        # If index not found, create a new one
        logging.info("indexes not found. Creating new ones...")
        
        #load data
        pfizer_report = SimpleDirectoryReader(input_files=["reports/pfizer_sec_filings_10k_2022.pdf"], filename_as_id=True).load_data()
        print(f"loaded pfizer sec filings 10k with {len(pfizer_report)} pages")

        merck_report = SimpleDirectoryReader(input_files=["reports/merck_sec_filings_10k_2022.pdf"], filename_as_id=True).load_data()
        print(f"loaded merck sec filings 10k with {len(merck_report)} pages")

        #build indices
        pfizer_index = VectorStoreIndex.from_documents(pfizer_report, service_context=service_context)
        print(f"built index for pfizer report with {len(pfizer_index.docstore.docs)} nodes")

        merck_index = VectorStoreIndex.from_documents(merck_report, service_context=service_context)
        print(f"built index for merck report with {len(merck_index.docstore.docs)} nodes")

        # persist both indexes to disk
        pfizer_index.storage_context.persist(persist_dir="./storage_pfizer")
        merck_index.storage_context.persist(persist_dir="./storage_merck")

        # update the global variables of pfizer_index_id and merck_index_id
        pfizer_index_id = pfizer_index.index_id
        merck_index_id = merck_index.index_id

        # save the variables to the file
        save_variables()

    #build query engines
    pfizer_report_engine = pfizer_index.as_query_engine(similarity_top_k=3)
    merck_report_engine = merck_index.as_query_engine(similarity_top_k=3)

    #build query engine tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine = pfizer_report_engine,
            metadata = ToolMetadata(name='pfizer_report_2022', description='Provides information on Pfizer SEC 10K filings for 2022')
        ),
        QueryEngineTool(
            query_engine = merck_report_engine,
            metadata = ToolMetadata(name='merck_report_2022', description='Provides information on Merck SEC 10K filings for 2022')
        )
    ]

    #define SubQuestionQueryEngine
    sub_question_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, service_context=service_context)
    
    # query unstructured question
    response = sub_question_engine.query(unstructured_question)
    st.write(str(response))


st.title("SEC 10-K Filings Analysis for Life Science Companies")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
selected_query = st.sidebar.radio("Select a Query:", ("Structured Data Query", "Unstructured Data Query"))

# Page for structured data query
if selected_query == "Structured Data Query":
    st.subheader("Structured Data Query")
    structured_query = st.text_area("Enter your question for structured Cybersyn SEC 10-K filings here", key="structured")
    if st.button("Ask"):
        if structured_query:
            structured_data_querying(structured_query)

# Page for unstructured data query
elif selected_query == "Unstructured Data Query":
    st.subheader("Unstructured Data Query")
    unstructured_query = st.text_area("Enter your question for unstructured SEC 10-K filings for Pfizer and Merck here", key="unstructured")
    if st.button("Ask"):
        if unstructured_query:
            # load the variables at app startup
            load_variables()
            unstructured_data_querying(unstructured_query)
