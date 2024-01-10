import streamlit as st
import numpy as np
import pandas as pd

import openai

import requests

api_key = ''
openai.api_key = api_key

# Define a function to interact with ChatGPT
def ask_chat_gpt(question):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=150  # Adjust the number of tokens as needed
    )
    return response.choices[0].text

st.title("Super Chat (Tableland and the graph version)")

def getData(query, tableName):
    url = 'http://localhost:3000/query-tableland'

    if tableName == "graph":
        ulr = ' http://localhost:3000/query-graph'

    # Define the data you want to send as JSON
    data = {'query': query}

    # Send a POST request with JSON data
    response = requests.post(url, json=data)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        return pd.DataFrame(response_data)
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    res = ask_chat_gpt(
        "you are a data scientist, please make only a json response with the table_name, the x and y for the graph and the query of this:" + prompt)
    with st.chat_message("assistant"):
        st.markdown("sure, here is the data:")
        data = getData(res.query, res.table_land)
        st.bar_chart(data.set_index(res.x)[res.y])

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": "sure, here is the data:"})