import streamlit as st
import openai
import time
import json
import sys
import os
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


st.title("eCommerce Database Agent Bot")


def init_config():
    if "db_chain" not in st.session_state:
        sys.path.insert(0, "../../0_common_config")
        from config_data import (
            get_deployment_name_turbo,
            set_environment_details_turbo,
            get_environment_details_turbo,
            get_lakehouse_connection_details,
        )

        st.session_state["deployment_name"] = get_deployment_name_turbo()
        deployment_id = get_deployment_name_turbo()
        set_environment_details_turbo()
        print(
            "deployment_name",
            st.session_state["deployment_name"],
            "\nopenai.api_base",
            openai.api_base,
            "\nopenai.api_type",
            openai.api_type,
        )
        key, base_uri, api_type, version = get_environment_details_turbo()
        llm = ChatOpenAI(deployment_id=deployment_id, temperature=0, openai_api_key=key)
        (
            az_synapse_db_server,
            az_synapse_db_name,
            az_synapse_db_user_name,
            az_synapse_db_password,
        ) = get_lakehouse_connection_details()

        db = SQLDatabase.from_uri(
            "mssql+pyodbc://"
            + az_synapse_db_user_name
            + ":"
            + az_synapse_db_password
            + "@"
            + az_synapse_db_server
            + ",1433/"
            + az_synapse_db_name
            + "?driver=ODBC+DRIVER+18+for+SQL+Server"
        )

        db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
        st.session_state["db_chain"] = db_chain
    return st.session_state["db_chain"]


# with open('claim-document.txt', 'r',encoding="utf8") as file:
#     case_data = file.read()


# with open('metaprompt.txt', 'r') as file:
#     # system_prompt clea= file.read().replace('\n', '')
#     system_prompt += file.read()
#     system_prompt += '\n'
# system_prompt += case_data

system_prompt = "You are an AI Assistant helping with users query for ecommerce data from a Lakehouse"
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": system_prompt})

counter = 0
for message in st.session_state.messages:
    if counter == 0:
        counter += 1
        st.text_area(label="system prompt", value=system_prompt)
        continue
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What products would you like to look for?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        db_chain = init_config()
        response = db_chain.run(prompt)
        print("*******************", response)
        # try:
        #     full_response += response.choices[0].delta.get("content", "")
        #     time.sleep(0.05)
        #     message_placeholder.markdown(full_response + "▌")
        # except:
        #     pass
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
