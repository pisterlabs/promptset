import streamlit as st
from streamlit_chat import message
from langchain import PromptTemplate
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
import pandas as pd
import numpy as np
import os
from google.oauth2 import service_account
import pandas_gbq
from pandas.io import gbq
import csv
from google.cloud import bigquery
import re
from pathlib import Path

st.set_page_config(
    page_title='Model Assistant',
)

# GBQ Credentials
# Credentials
home = str(Path.home())
credential_path = home + r'\Waternlife\05_Business Intelligence - General\06_BI Team Documents\09_Important docs\01_API KEYS - PROTECTED\giovanni_keys\danish-endurance-analytics-3cc957295117.json'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
credentials = service_account.Credentials.from_service_account_file(
    credential_path,
)

# Construct Big Query Clientc
client = bigquery.Client()

# SESSION STATES
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'disabled' not in st.session_state:
    st.session_state.disabled = {'ui_models': False, 'ui_query': True}

if 'visibility' not in st.session_state:
    st.session_state.visibility = {'ui_models': True, 'ui_query': True}

if 'ui_inputs' not in st.session_state:
    st.session_state.ui_inputs = {'ui_models': '', 'ui_query': ''}

# CALLBACKS HANDLERS
def disable_uimodels():
    st.session_state.disabled['ui_models'] = True
    st.session_state.disabled['ui_query'] = False

def disable_uiquery():
    st.session_state.disabled['ui_query'] = True

def hide_uiinputs():
    st.session_state.visibility['ui_models'] = False
    st.session_state.visibility['ui_query'] = False
    ui_models_empty.empty()
    ui_query_empty.empty()

# GBQ Functions
def get_table_schema(client, table_id):
    if "`" in table_id:
        table_id = table_id.replace("`", "")
    table_ref = client.get_table(table_id)
    schema_str = ""
    schema_info = {}
    for field in table_ref.schema:
        schema_str += f"{field.name} ({field.field_type}), "
        schema_info[field.name] = field.field_type

    return schema_str[:-1], schema_info

# LLM Functions
def get_prompt_formated(ui_models, ui_query, ui_columns):
    from utils_data_engineering import get_table_schema
    template = '''
    {ui_query}

    Take in to consideration this GBQ model "{ui_models}" and its respective schema containing its metadata and data types, carefully pay attention in the user input below, and help me to construct a new model based on the user input and models metadata.

    {ui_models}'s Columns and Data Types are:
    {ui_columns}
    '''
    prompt = PromptTemplate(
        input_variables=['ui_models', 'ui_columns', 'ui_query'],
        template=template
    )

    prompt_formatted = prompt.format(
        ui_models=ui_models,
        ui_columns=ui_columns,
        ui_query=ui_query
    )
    return prompt_formatted


ui_models_empty = st.empty()
ui_query_empty = st.empty()

if st.session_state.visibility['ui_models']:
    st.session_state.ui_inputs['ui_models'] = ui_models_empty.text_input("Wich models do you want to work with?",
                               disabled=st.session_state.disabled['ui_models'],
                               on_change=disable_uimodels)
    

if st.session_state.visibility['ui_query']:
    st.session_state.ui_inputs['ui_query'] = ui_query_empty.text_input("What is your query?",
                              disabled=st.session_state.disabled['ui_query'],
                              on_change=disable_uiquery)
    
if st.session_state.ui_inputs['ui_models'] != '' and st.session_state.ui_inputs['ui_query'] != '':
    # Create placeholder objects for the columns and the submit button
    col_datetime_placeholder = st.empty()
    col_float_placeholder = st.empty()
    col_string_placeholder = st.empty()
    submit_button_placeholder = st.empty()  # Placeholder for the submit button

    # Initialize a dictionary to store the state of each checkbox
    selected_fields = {}

    if 'schema_info' not in st.session_state:
        st.session_state.schema_info = {}
        st.session_state.schema_info = get_table_schema(client=client, table_id=st.session_state.ui_inputs['ui_models'])[1]
        schema_info = st.session_state.schema_info
    else:
        schema_info = st.session_state.schema_info

    # If the placeholders are not cleared, display the checkboxes
    if 'clear_columns' not in st.session_state or not st.session_state.clear_columns:
        with col_datetime_placeholder.container():
            col_datetime, col_float, col_string = st.columns(3)
            col_datetime.subheader('Date')
            col_float.subheader('Numeric')
            col_string.subheader('Categoric')

            for field, field_type in schema_info.items():
                checkbox_key = f"{field}_{field_type.lower()}"
                with (col_datetime if field_type in ['DATETIME', 'TIMESTAMP'] else 
                    col_float if field_type in ['FLOAT', 'INTEGER'] else 
                    col_string):
                    # If checkbox is ticked, store the field with its type in selected_fields
                    if st.checkbox(field, key=checkbox_key):
                        selected_fields[field] = {'data_type': field_type}

    # Use the submit button placeholder to display the button
    submit_button = submit_button_placeholder.button('Submit')
    if submit_button:
        col_datetime_placeholder.empty()
        col_float_placeholder.empty()
        col_string_placeholder.empty()
        submit_button_placeholder.empty()  # Hide the submit button

        # Set the flag to clear the columns
        st.session_state.clear_columns = True

# # creating the sidebar
with st.sidebar:
    st.write('### Model Assistant')
    if st.session_state.ui_inputs['ui_models'] != '' and st.session_state.ui_inputs['ui_query'] != '':
        formatted_models = "\n\n".join(st.session_state.ui_inputs['ui_models'].split(","))
        # remove danish-endurance-analytics. from formamtted_models
        formatted_models = re.sub(r'danish-endurance-analytics.', '', formatted_models)
        formatted_models_with_backticks = f"```\n{formatted_models}\n```"
        st.markdown(f'**Selected Model:**\n\n{formatted_models_with_backticks}')
        # Hide UI Inputs
        hide_uiinputs()
        if submit_button:
                # Display which fields have been selected
            st.write('**Selected Columns:**')
            for field, attributes in selected_fields.items():
                # Extract data_type from the attributes
                data_type = attributes['data_type']
                # Display the field and data type in a code-like format
                st.markdown(f"- `{field} - {data_type}`")


# instatianting the chat model
chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.5, openai_api_key="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")

#System Message
if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
    # read system_messages.csv file into a string
    with open('model_priming/system_message.txt', 'r') as f:
        system_message = f.read()
    st.session_state.messages.append(
        SystemMessage(content=system_message)
        )


# if ui models and ui query are not empty
if st.session_state.ui_inputs['ui_models'] != '' and st.session_state.ui_inputs['ui_query'] != '' and submit_button:
    def get_selected_fields_formatted(selected_fields):
        formatted_fields = []
        for field, attributes in selected_fields.items():
            # Extract data_type from the attributes
            data_type = attributes['data_type']
            # Format the field and data type and append to the list
            formatted_fields.append(f"{field}({data_type})")
        # Join all formatted fields into one string, separated by commas
        selected_fields_formatted = ", ".join(formatted_fields)
        return selected_fields_formatted

    # first prompt
    # if only system message is in there then append the first prompt
    if len(st.session_state.messages) == 1:
        st.session_state.messages.append(HumanMessage(
                content=get_prompt_formated(
                    ui_models=st.session_state.ui_inputs['ui_models'],
                    ui_query=st.session_state.ui_inputs['ui_query'], ui_columns=get_selected_fields_formatted(selected_fields))))
        
        with st.spinner('Working on your request ...'):
            # creating the ChatGPT response
            response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))


    prompt = st.chat_input("What is up?")
    # If the user entered a question, append it to the session state
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.spinner('Working on your request ...'):
            # creating the ChatGPT response
            response = chat(st.session_state.messages)

        # adding the response's content to the session state
        st.session_state.messages.append(AIMessage(content=response.content))


    # displaying the messages (chat history)
    for i, msg in enumerate(st.session_state.messages[1:]):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assitant"):
                st.markdown(msg.content)

# with st.sidebar:
#     st.session_state.messages