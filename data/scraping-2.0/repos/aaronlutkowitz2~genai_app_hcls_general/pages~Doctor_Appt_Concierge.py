# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation Date: July 10, 2023

@author: Aaron Wilkowitz
"""

################
### import 
################
# gcp
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import bigquery

import utils_config

# # others
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
# from langchain import LLM
from langchain.llms import VertexAI
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

# others
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import datetime
import re
# from datetime import datetime  
# from datetime import timedelta  
# import timedelta
import pandas as pd

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Doctor Schedule Appt')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Date**: 2023-08-22')
st.write('**Purpose**: Doctor concierge.')

# Gitlink
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('30 Second Video')

video_url = 'https://youtu.be/-l-lIOrz8Qc'
st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSG53Quz0A7HArZgZbgRwg-uIctSzoND869pjzTHKbgjVr_0riJS4n_8p4E2N4lW1mboLC-B8VxO1Ay/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### model inputs
################

# Model Inputs
model_id = 'text-bison@001' 
model_temperature = 0
model_token_limit = 200 
model_top_k = 1
model_top_p = 0

################
### Intro - select workflow
################

# Select workflow
st.divider()
st.header('1. Determine issue')

welcome_text_bolded = 'Welcome back, Sally!' 
welcome_text = 'Use this personal concierge app to assist with scheduling help for any of your medical needs'

st.write(f':blue[**{welcome_text_bolded}**] ' + welcome_text)

# Body Part
param_issue = st.selectbox(
    'What medical issue can we help you with?'
    , (
          'I just found out I am pregnant'
        , 'I think my arm is broken.'
        , 'My chest hurts'
        , 'custom'
      )
    , index = 0
  )

if "custom" in param_issue:
  custom_goal = st.text_input('Write your custom medical issue here')
  issue = custom_goal
else:
  issue = param_issue

st.write(':blue[**Issue:**] ' + issue)

llm_prompt_a = f'''

I am a patient with an issue: {issue}

What do I need to do?

'''

PROJECT_ID = utils_config.get_env_project_id()
LOCATION = utils_config.LOCATION

# Run the first model
vertexai.init(
      project = PROJECT_ID
    , location = LOCATION)
parameters = {
    "temperature": model_temperature,
    "max_output_tokens": model_token_limit,
    "top_p": model_top_p,
    "top_k": model_top_k
}
model = TextGenerationModel.from_pretrained(model_id)
response = model.predict(
    f'''{llm_prompt_a}''',
    **parameters
)
# print(f"Response from Model: {response.text}")

llm_response_text_a = response.text

st.write(':blue[**Response:**] ' + llm_response_text_a)

# Body Part
param_next_step1 = st.selectbox(
    'Do you need more help?'
    , (
          'Yes - I would like to schedule an appointment'
        , 'No - I am good, thank you.'
      )
    , index = 0
  )

if "No" in param_next_step1: 
    exit() 
else:
    pass 

################
### Determine doctor type
################

st.divider()
st.header('2. Determine doctor type')

llm_prompt_b = f'''
Create a comma separated list of the specific doctors that a patient needs to see based on the medical conditions described below:

\n\n

input: I just found out I am pregnant \n
output: Obstetrician/Gynecologist (OB/GYN), Primary Care Physician (PCP), Widwife \n

\n\n

input: I think my arm is broken. \n
output: Orthopedist, General Practitioner \n

\n\n

input: My chest hurts \n
output: Cardiologist, Pulmonologist, Gastroenterologist \n

input: {issue} \n
output: 

'''

vertexai.init(
    project = PROJECT_ID
    , location = LOCATION)
parameters = {
    "temperature": model_temperature,
    "max_output_tokens": model_token_limit,
    "top_p": model_top_p,
    "top_k": model_top_k
}
model = TextGenerationModel.from_pretrained(model_id)
response = model.predict(
    f'''{llm_prompt_b}''',
    **parameters
)
llm_response_text_b = response.text
llm_response_text_b_added = "I am not sure, " + llm_response_text_b

llm_response_text_b_list = llm_response_text_b_added.split(',')

param_doctor = st.selectbox(
'What kind of doctor would you like to see?'
, (
    llm_response_text_b_list
    )
, index = 0
)

if param_doctor == "I am not sure":
    llm_prompt_c = f'''
    I am deciding which kind of provider I should see. 

    Context: {issue}

    I am choosing between {llm_response_text_b}.

    What are the reasons to see each kind of provider?

    '''

    vertexai.init(
        project = PROJECT_ID
        , location = LOCATION)
    parameters = {
        "temperature": model_temperature,
        "max_output_tokens": model_token_limit,
        "top_p": model_top_p,
        "top_k": model_top_k
    }
    model = TextGenerationModel.from_pretrained(model_id)
    response = model.predict(
        f'''{llm_prompt_c}''',
        **parameters
    )
    llm_response_text_c = response.text

    st.write(':blue[**Additional Context:**] ')
    st.text(llm_response_text_c)

    llm_response_text_c_list = llm_response_text_b.split(',')

    param_doctor2 = st.selectbox(
    'Given that additional context, what kind of doctor would you like to see?'
    , (
        llm_response_text_c_list
        )
    , index = 0
    )

    doctor = param_doctor2

else: 
    doctor = param_doctor 
st.write(':blue[**Doctor:**] ' + doctor)

client_bq = bigquery.Client()

dataset_id = 'fake_hospital_data'
table_id = 'doctor_data_fake_typed'
table_info = f'`{PROJECT_ID}.{dataset_id}.{table_id}`'
sql = f"""
SELECT distinct doctor_type 
FROM {table_info} 
GROUP BY 1 
"""
doctor_df = client_bq.query(sql).to_dataframe()
doctor_list = doctor_df['doctor_type'].to_list() # values
doctor_string = '\n- '.join(doctor_list)

llm_prompt_d = f'''
I want to see a {doctor}. I need to fill out a form that only allows me to select ONE of the below options. Which of the below list most closely matches {doctor}? Only provide the exact line from the list below - do not add any other context. ONLY include one of the lines below

- {doctor_string}'''

model_token_limit_d = 20 
vertexai.init(
    project = PROJECT_ID
    , location = LOCATION)
parameters = {
    "temperature": model_temperature,
    "max_output_tokens": model_token_limit_d,
    "top_p": model_top_p,
    "top_k": model_top_k
}
model = TextGenerationModel.from_pretrained(model_id)
response = model.predict(
    f'''{llm_prompt_d}''',
    **parameters
)
llm_response_text_d = response.text

official_doctor_type = llm_response_text_d

st.write(':blue[**Official Doctor Title:**] ' + official_doctor_type)

################
### Determine hospital
################

st.divider()
st.header('3. Determine hospital')

zip_code = st.number_input(
    label = "Zip Code"
   , min_value = 501
   , max_value = 99999
   , value = 22046
)

if zip_code < 10000:
    hospital_code = 0
else: 
    hospital_code = int(str(zip_code)[0])

if hospital_code == 0:
    hospital_name = 'Aurora Medical Institute'
elif hospital_code == 1:
    hospital_name = 'Crestview Wellness Center'
elif hospital_code == 2:
    hospital_name = 'Crystal Lake Medical Center'
elif hospital_code == 3:
    hospital_name = 'Evergreen Medical Center'
elif hospital_code == 4:
    hospital_name = 'Horizon Care Clinic'
elif hospital_code == 5:
    hospital_name = 'Maplewood General Hospital'
elif hospital_code == 6:
    hospital_name = 'Meadowbrook Community Hospital'
elif hospital_code == 7:
    hospital_name = 'Serenity Health Hospital'
elif hospital_code == 8:
    hospital_name = 'Sunflower Health Pavilion'
elif hospital_code == 9:
    hospital_name = 'Willowbrook Regional Clinic'
else: 
    hospital_name = 'Crestview Wellness Center'

st.write(':blue[**Your Nearest Hospital:**] ' + hospital_name)

################
### Determine doctor
################

st.divider()
st.header('4. Determine doctor')

additional_context = st.text_input(
   'If you have any additional preferences on your doctor (e.g. gender, language), provide them here'
   , value = ""
  )

if additional_context == "":
    additional_context_text = ""
else: 
    additional_context_text = f"""
Additional Context: 
- The patient wants: {additional_context}
"""

input = "Can you provide a list of 5 doctors who meet the criteria above?"

sql_prompt = f"""
You are a GoogleSQL expert. Given an input question, first create a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {{top_k}} results using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the following tables:
{table_id}

Rule 1:
Do not filter and query for columns that do not exist in the table being queried.

Rule 2:
Only use the columns in this table: 
SELECT column_name
FROM `{PROJECT_ID}.{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
WHERE table_name = "{table_id}"

Rule 3:
The patient wants to see a {official_doctor_type} doctor type.

Rule 4:
The patient wants to see a doctor at '{hospital_name}' hospital

Rule 5:
Follow each rule every time.

{additional_context_text}

Question: {input}
"""

table_names = [table_id]
table_uri = f"bigquery://{PROJECT_ID}/{dataset_id}"
engine = create_engine(f"bigquery://{PROJECT_ID}/{dataset_id}")


# LLM model
model_name = "text-bison@001" #@param {type: "string"}
max_output_tokens = 1024 #@param {type: "integer"}
temperature = 0.2 #@param {type: "number"}
top_p = 0.8 #@param {type: "number"}
top_k = 40 #@param {type: "number"}
verbose = True #@param {type: "boolean"}

llm = VertexAI(
  model_name=model_name,
  max_output_tokens=max_output_tokens,
  temperature=temperature,
  top_p=top_p,
  top_k=top_k,
  verbose=verbose
)

#create SQLDatabase instance from BQ engine
db = SQLDatabase(
    engine=engine
    ,metadata=MetaData(bind=engine)
    ,include_tables=table_names # [x for x in table_names]
)

#create SQL DB Chain with the initialized LLM and above SQLDB instance
db_chain = SQLDatabaseChain.from_llm(
    llm
    , db
    , verbose=True
    , return_intermediate_steps=True)

#Define prompt for BigQuery SQL
_googlesql_prompt = sql_prompt

GOOGLESQL_PROMPT = PromptTemplate(
    input_variables=[
    #    "input"
    #  , "table_info"
        "top_k"
    #  , "project_id"
    #  , "dataset_id"
    ],
    template=_googlesql_prompt,
)

#passing question to the prompt template
final_prompt = GOOGLESQL_PROMPT.format(
#    input=question
#  , project_id =project_id
#  , dataset_id=dataset_id
#  , table_info=table_names
    top_k=10000
)

# pass final prompt to SQL Chain
output = db_chain(final_prompt)

# outputs
st.write(':blue[**SQL:**] ')
sql_answer = output['intermediate_steps'][1]
st.code(sql_answer, language="sql", line_numbers=False)

output_result = output['result']
st.write(':blue[**Answer:**] ')
st.write(output_result)
output_results_list = output_result.split(',')

# st.write(':blue[**Full Work:**] ')
# st.write(output)  

param_doctor_select = st.selectbox(
'Which doctor would you like to see?'
, (
    output_results_list
    )
, index = 0
)

################
### Schedule Appt
################

st.divider()
st.header('5. Schedule Appointment')

today = datetime.datetime.today()
today_date = datetime.date.today()
dow = today_date.weekday()

if dow == 0: # Monday
    adjustment1 = 1 
    adjustment2 = 4
elif dow == 1: # Tuesday
    adjustment1 = 1 
    adjustment2 = 3
elif dow == 2: # Wednesday
    adjustment1 = 5 
    adjustment2 = 9
elif dow == 3: # Thursday
    adjustment1 = 4 
    adjustment2 = 8
elif dow == 4: # Friday
    adjustment1 = 3 
    adjustment2 = 7
elif dow == 5: # Saturday
    adjustment1 = 2 
    adjustment2 = 6
elif dow == 6: # Sunday
    adjustment1 = 1 
    adjustment2 = 5
else:
    adjustment1 = 1 
    adjustment2 = 4

datetime_future1 = today + datetime.timedelta(days=adjustment1)
datetime_future1b = datetime_future1 + datetime.timedelta(days=1)
datetime_future2 = today + datetime.timedelta(days=adjustment2)
date_future1 = datetime_future1.date()
date_future2 = datetime_future2.date() 
date = st.date_input(
     label = "When would you like to schedule your appointment?"
   , value = datetime_future1b
   , min_value = date_future1
   , max_value = date_future2
)

time = st.time_input(
      label = "What time would you like to schedule your appointment?"
    , value = datetime.time(10, 30)
    , step = 1800
)

success_text = f'Your appointment with {param_doctor_select} has been scheduled for {date} at {time}'
st.write(':green[**Success:**] ' + success_text)

