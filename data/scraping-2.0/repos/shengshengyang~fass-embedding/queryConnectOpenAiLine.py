import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
import os
import json
from dotenv import load_dotenv
import time
import openai

# Set the page to wide layout
st.set_page_config(layout="wide")
# Read data from phone.xlsx
data = pd.read_excel('phone1.xlsx')
#load key
openai.api_key = os.getenv("OPENAI_KEY")

# Load the Faiss index and title vectors from files
index = faiss.read_index('index.faiss')
title_vectors = np.load('title_vectors.npy')
# Maximum number of tokens for a conversation
max_tokens = 4096

# Set up Streamlit app
st.markdown("<h1 style='text-align: center; color: black;'>大豐智慧課服</h1>", unsafe_allow_html=True)
# Create two columns
col1, col2 = st.columns(2)
load_dotenv()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Create a placeholder for "GPT生成回覆" section
gpt_placeholder = col2.empty()
query = col2.text_input("請輸入您的問題，我將為您回答")
col2.subheader("GPT生成回覆:")
col1.subheader("最符合您問題的五個答案:")

def summarize_chat_history(chat_history):
    if not chat_history:
        return ""
    # Convert chat history to text
    chat_text = " ".join([message["content"] for message in chat_history])

    total_words = len(chat_text.split())

    if total_words > 100:
        # If the total number of words exceeds 100, summarize the chat text using OpenAI's API
        response = openai.Completion.create(
            engine="gpt-4",  # Change this to the correct engine name when GPT-4 is available
            prompt=chat_text,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    else:
        # If the total number of words is less than or equal to 100, return the original chat text
        return chat_text
def count_tokens(messages):
    return sum([len(message["content"].split()) for message in messages])

# Before calling the openai.ChatCompletion.create method
chat_summary = summarize_chat_history(st.session_state.messages)
if query:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with col2.chat_message(message["role"]):
            st.markdown(message["content"])
    with col2.chat_message("user"):
        st.markdown(query)
        # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Use the OpenAI library to make the API call
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query]
    )
    query_vector = np.array(response.data[0]['embedding'])

    # Search for nearest neighbors
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(np.array([query_vector]), k)

    # Retrieve the matched content
    matched_data = data.iloc[indices[0]]


    def replace_none_with_na(value):
        return '無' if pd.isnull(value) else value
        # Create two columns


    for i, row in matched_data.iterrows():
        row = row.apply(replace_none_with_na)
        html = """
        <div style="border:1px solid #000; margin:10px; padding:10px;">
            <h5>問題: {question}</h5>
            <p>解答: {answer}</p>
        </div>
        """.format(question=row['question'], answer=row['answer'])
        col1.markdown(html, unsafe_allow_html=True)

    top_results_str = json.dumps(json.loads(matched_data.to_json(orient='records')), ensure_ascii=False)

    api_endpoint = "https://api.openai.com/v1/chat/completions"
    messages = [{"role": "system", "content": "你是一位大豐環保公司的客服人員，語言使用繁體中文，服務地區包括 基隆、台北、大台北地區、高雄、花蓮、雲林、南投、桃園、嘉義、新竹、台南、宜蘭、台東、苗栗、台中、大台中地區、彰化、屏東 ,"
                                              "客服電話: 0800588717"}] \
               + st.session_state.messages \
               + [
                   {"role": "assistant", "content": "以下為參考資料: " + top_results_str},
                  {"role": "user", "content": "請根據提供的question欄位，利用answer 的資料，以及歷史訊息回答問題: " + query
                                              + ",若資料沒有能夠回答問題請以下列字句回復: 目前尚無相關資料，可留下您的手機號碼，我們將為您服務 "},
                  ]
    print(messages)
    print('---------------------------------------')
    # Count the number of tokens in the messages
    num_tokens = count_tokens(messages)

    # If the number of tokens exceeds the maximum, remove the oldest messages until it doesn't
    while num_tokens > max_tokens:
        removed_message = st.session_state.messages.pop(0)
        num_tokens -= len(removed_message["content"].split())
    # Generate response using ChatGPT API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )


    if 'choices' in response:
        # Extract the generated response
        generated_response = response.choices[0].message.content

        with col2.chat_message("assistant"):
            # Display the full response immediately
            st.markdown(generated_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": generated_response})
    else:
        print("No response choices found.")
