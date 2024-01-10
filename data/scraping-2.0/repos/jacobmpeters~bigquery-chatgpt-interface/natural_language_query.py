import os
import openai
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from google.cloud import bigquery
import db_dtypes
import json

# import pandas_gbq

def get_data_sample(bq_client, bq_table, row_limit=5):
  '''Retrieves a sample of data from a BigQuery table.'''
  full_table_name = f'{bq_table.project}.{bq_table.dataset_id}.{bq_table.table_id}'
  sql = f'''SELECT * FROM `{full_table_name}` LIMIT {row_limit}'''
  df = bq_client.query(sql).to_dataframe()
  return df

def construct_initial_gpt_prompt(bq_client: bigquery.Client, bq_table: bigquery.TableReference) -> str:
    sample_df = get_data_sample(bq_client, bq_table, row_limit=5)

    prompt = f'''Act as if you're a data scientist who uses exclusively GoogleSQL syntax in BigQuery. 
Note that in 2021, GoogleSQL was called Google Standard SQL.

You have a BigQuery table named {bq_table.full_table_id} with the following schema:
```{bq_table.schema}```

The first rows look like this: 
```{sample_df}```

Based on this data, write a SQL query to answer my questions.
Return the SQL query ONLY so that it will be executable in BigQuery.
Do not include any additional explanation.
Remember that table names must be in the form of `project.dataset_id.table_id` in a GoogleSQL query.
'''
    return prompt
  

def construct_gpt_eror_prompt(sql, bq_errors):
  prompt = f'''I ran the query in BigQuery and received the following error(s):
            ```{bq_errors}```
            Return a corrected SQL query only with no aditional explanation.
             '''
  return prompt


def main(full_table_name, nat_lang_query):
  
  # Initialize BigQuery client and table object
  bq_client = bigquery.Client()
  bq_table  = bq_client.get_table(full_table_name)
  
  # Initialize openai API Client
  openai.api_key = os.getenv('OPENAI_KEY') # this is the client object

  # Construct the initial GPT prompt
  initial_prompt = construct_initial_gpt_prompt(bq_client, bq_table)
  
  # Initialize the list of messages with a system message and the initial user message
  messages = [
    {'role': 'system', 'content': initial_prompt},
    {'role': 'user',   'content': nat_lang_query}]
  
  # Make the initial OpenAI API call
  response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo', 
    messages=messages)
  
  # Extract and add the new message from the API response
  new_message = response['choices'][0]['message']
  messages.append({'role': 'assistant', 'content': new_message['content']})
  sql = new_message['content']
  # print(sql)
  
  # Query BQ until successful bq_result or max attempts reached

  max_attempts = 10
  for i in range(0, max_attempts-1):
  
    # Query BQ table
    bq_result = bq_client.query(sql)
    
    # Handle errors
    bq_errors = bq_result.errors
    
    if bq_errors:
      error_prompt = construct_gpt_eror_prompt(sql, bq_errors)
      messages.append({'role': 'user', 'content': error_prompt})
      response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo', 
        messages=messages)
      
      # Extract and add the new message from the API response
      new_message = response['choices'][0]['message']
      messages.append({'role': 'assistant', 'content': new_message['content']})
      sql = new_message['content']
      bq_result = bq_client.query(sql)
    else:
      break

  # View bq_results 
  return bq_result.to_dataframe(), sql, messages

    
if __name__ == "__main__":
  
    # Table name in form of project.datset.table
  full_table_name = 'bigquery-public-data.usa_names.usa_1910_2013'
  #full_table_name = 'bigquery-public-data.covid19_nyt.excess_deaths'
  
  # Natural language query
  # nat_lang_query  = 'Get the top 5 baby names in 2013. Include the column headers name, year, and occurances in the output.'
  nat_lang_query  = 'Get the top 5 baby names in 2013 as a string.' # this one receives/fixes BQ errors
  # nat_lang_query = 'What are the top 5 baby names in 2012 that begin with the letter A?'
  # nat_lang_query = 'How many excess deaths occured in 2022 in Spain due to Covid-19?'
  
  bq_result, sql, messages = main(full_table_name, nat_lang_query)
  
  # Format output
  bq_result_markdown = bq_result.to_markdown(index=False)
  output_md = f'''
  ### Natural Language Query:\n\n{nat_lang_query}\n\n 
  ### SQL From GPT:\n\n```sql\n{sql}\n```\n
  ### BigQuery Results:\n\n{bq_result_markdown}\n\n'''
  print(output_md)
  
  formatted_messages = [json.dumps(message, indent=4) for message in messages]
  formatted_messages = "\n".join(formatted_messages)
  messages_md = "```JSON\n" + formatted_messages + "\n```"
  print('### GPT Messages:\n\n' + messages_md)
  

  