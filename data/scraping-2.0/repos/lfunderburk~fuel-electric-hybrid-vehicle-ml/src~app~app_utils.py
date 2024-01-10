import pandas as pd
import os
import openai
from sqlalchemy.engine import create_engine
from pathlib import Path
from pydantic import BaseModel

class Query(BaseModel):
    query: str

class Prompter:
    def __init__(self, api_key, gpt_model):
        if not api_key:
            raise Exception("Please provide the OpenAI API key")


        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.gpt_model = gpt_model
    
    def prompt_model_return(self, messages: list):
        response = openai.ChatCompletion.create(model=self.gpt_model, 
                                                messages=messages,
                                                temperature=0.2)
        return response["choices"][0]["message"]["content"]
    


def init_prompt(query: Query, prompter: Prompter, df: pd.DataFrame, sample_values: dict):
  # Generate SQL query
  system_content = "You are a data analyst specializing in SQL, \
                  you are presented with a natural language query, \
                  and you form queries to answer questions about the data."
  user_content = f"Please generate 1 SQL queries for data with \
                  columns {', '.join(df.columns)} \
                  and sample values {sample_values}. \
                   The table is called 'vehicleDB'. \
                   Use the natural language query {query.query}"
  datagen_prompts = [
                {"role" : "system", "content" : system_content},
                {"role" : "user", "content" : user_content},
                  ]

  # Take parameters and form a SQL query
  sql_result= prompter.prompt_model_return(datagen_prompts)

  # Sometimes the query is verbose - adding unnecessary explanations
  sql_query = sql_result.split("\n\n")[0]

  return sql_query

def init_data():
    # Set the path to the raw data
    # Convert the current working directory to a Path object
    script_dir = Path(os.getcwd())
    predicted_data_path = script_dir / 'data' / 'predicted-data' / 'vehicle_data_with_clusters.csv'
    
    # Load the CSV file into a DataFrame
    dirty_df = pd.read_csv(predicted_data_path)
    global df
    df = data_cleaner(dirty_df)
    global sample_values
    sample_values = {df.columns[i]: df.values[0][i] for i in range(len(df.columns))}

    return df, sample_values
    
def data_cleaner(df):
    

    # Replace the characters '.', '/', '(' and ')'with '_per_' all entries
    df.columns = df.columns.str.replace('.', '_', regex=True)
    df.columns = df.columns.str.replace('/', '_per_', regex=True)
    df.columns = df.columns.str.replace('(', '_', regex=True)
    df.columns = df.columns.str.replace(')', '_', regex=True)

    # drop column hybrid_in_fuel	hybrid_in_electric	aggregate_levels	vehicle_type_cat
    df = df.drop(['hybrid_in_fuel', 'hybrid_in_electric', 'aggregate_levels','transmission_','fuel_type'], axis=1)

    return df

def init_database(df):
    # Set up engine
    engine = create_engine("sqlite://")
    df.to_sql("vehicleDB", engine)

    return engine

