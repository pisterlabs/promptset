#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import openai
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

def text_to_sql(table_name,text,column_names):
    
    #text to sql 
    #generate_questions_with_answers from chatgpt API
    instruction1 = "turn this phrase into a sql query: "
    instruction2 = f"the table name is'{table_name}', the columns names are '{column_names}'"
    openai.api_key = 'Your_key'
    # Define the instruction and prompt for the model
    prompt = f"{instruction1}{text}{instruction2}"

    # Generate questions and answer options using the ChatGPT model
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        n=1,  # Generate three questions
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    #full text for the form
    sql_query = response["choices"][0]["text"]

    print("finished generating sql_query")
    return sql_query

