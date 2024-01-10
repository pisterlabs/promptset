from dotenv import load_dotenv
import os
import pandas as pd
import openai
from execution_manager import run_sql

def set_API_key():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key

def set_database_connection(): 
    load_dotenv()
    database_url = f"postgresql://{os.getenv('USER')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}:{os.getenv('PORT')}/{os.getenv('DATABASE')}"
    return database_url

def set_schema(database_url: str):
    schema_sql = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema NOT IN ('pg_catalog', 'information_schema');
    """
    schema_df = run_sql(schema_sql, database_url)
    return schema_df

def add_synonyms_if_available(schema_df):
    if os.getenv("SYNONYM_TABLE_FILE_PATH") and os.path.exists(os.getenv("SYNONYM_TABLE_FILE_PATH")):
        schema_and_synonyms_df = add_synonyms(schema_df)
    else:
        schema_and_synonyms_df = schema_df
    return schema_and_synonyms_df

def add_synonyms(schema_df):
    synonym_table_file_path = os.getenv("SYNONYM_TABLE_FILE_PATH")
    synonyms_df = pd.read_csv(synonym_table_file_path)
    schema_and_synonyms_df = pd.merge(schema_df, synonyms_df, how='left',
                     left_on='column_name', right_on='existing_column_name')
    # Drop the 'existing_column_name' column as it's redundant
    schema_and_synonyms_df.drop('existing_column_name', axis=1, inplace=True)
    return schema_and_synonyms_df
