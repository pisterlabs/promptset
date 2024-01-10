from sqlalchemy import create_engine, inspect, MetaData, DDL
import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, GPTSimpleKeywordTableIndex
from llama_index import StorageContext, load_index_from_storage
import my_key
from llama_index import ServiceContext, LLMPredictor
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd

data_folder = "data"

def get_create_table_statement(engine, table_name):
    with engine.connect() as connection:
        # Get the DDL object for the CREATE TABLE statement
        statement = DDL(f"SHOW CREATE TABLE {table_name};")
        result = connection.execute(statement)
        create_table_statement = result.fetchone()[1]
    return create_table_statement

def run_sql(db_url, query):
    engine = create_engine(db_url, pool_recycle=3600)
    conn = engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    # Close the connection
    cursor.close()
    conn.close()
    engine.dispose()
    # Convert the result to a DataFrame
    df = pd.DataFrame(result, columns=column_names)
    return df

def scan_table_to_file(db_url):
    engine = create_engine(db_url)
    inspector = inspect(engine)
    metadata = MetaData()
    table_names = inspector.get_table_names()

    for table_name in table_names:
        create_table_statement = get_create_table_statement(engine, table_name)
        
        with open(f"data/{table_name}.sql", "w", encoding='utf-8') as file:
            file.write(create_table_statement)
    engine.dispose()

def build_index(selected_files):
    # Set up LLM
    llm = ChatOpenAI(model=model_name, max_tokens=2500)
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    # Generate the local index
    selected_files_with_path = [f"{data_folder}/{file_name}" for file_name in selected_files]
    documents = SimpleDirectoryReader(input_files=selected_files_with_path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist('db_index')

def create_query(query_str, request_str):
    llm = ChatOpenAI(model=model_name, max_tokens=2500, temperature=0.7)
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults(persist_dir="./db_index")
    db_index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = db_index.as_query_engine()
    response = query_engine.query("You are a MySQL query generation bot. Please provide a query command based on the table names, column names, and requirements of the current MySQL database. Query string: (%s); Request string: (%s)" % (query_str, request_str))
    return response

def analyse_db(db_question):
    llm = ChatOpenAI(model=model4_name, temperature=0.7, max_tokens=6000)
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults(persist_dir="./db_index")
    db_index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = db_index.as_query_engine()
    response = query_engine.query("You are a senior MySQL data warehouse engineer. Please answer the following question: %s. Please focus on the key points and keep the response within 4000 Chinese characters, concise, persuasive, and informative." % db_question)
    return response
