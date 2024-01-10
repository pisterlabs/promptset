import streamlit as st
import streamlit_authenticator as stauth
import openai
import requests
from requests.auth import HTTPBasicAuth
import json
import pandas as pd
import datetime as dt
import tiktoken
import os



#Authentication
import yaml
from yaml.loader import SafeLoader
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')


if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')

    st.title("Chat with your sleep data")
    
    token = os.getenv("OURATOKEN")
    start_date = "2023-10-01"
    end_date = "2023-12-01"

    # Set OpenAI API key from Streamlit secrets
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def get_sleep_data(token, start_date, end_date):
        url = "https://api.ouraring.com/v2/usercollection/sleep"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        params = {
            "start_date": start_date,
            "end_date": end_date
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return None
        
    # get data
    

    data = get_sleep_data(token, start_date, end_date)
    st.write("got data")
    
    from openai import OpenAI
    
    client = OpenAI()

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-1106-preview"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask questions to your sleep data?"):
        st.session_state.messages.append({"role": "system", "content": f"Here is a json with sleep data from a 35 year old age group triathlete. Help him to sleep better {data}"})
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
