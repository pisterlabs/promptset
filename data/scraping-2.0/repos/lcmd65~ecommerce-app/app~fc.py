import openai
import streamlit as st
from PIL import Image
from streamlitextras.webutils import stxs_javascript
from typing import NoReturn
from pymongo import MongoClient
import json
import os
import warnings
warnings.filterwarnings("ignore")

def redirect(url: str="http://localhost:8081/") -> NoReturn:
    stxs_javascript(f"window.location.href='{url}';")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    with open("app/schema.json", "r") as file:
        uri = json.load(file)
    client = MongoClient(uri["mongo_uri"])
    client.admin.command('ping')
    db = client["Api"]
    collection = db["api"]
    documents = collection.find()
    api_key = None
    for item in documents:
        if item['api'] == 'datathon-service':
            api_key = item['api-key']
            break
    client.close()
    openai_api_key = api_key

st.set_page_config(page_title="SUSBot", page_icon="app/static/images/logo.png")

st.button("Back", on_click=redirect)

col1, col2 = st.columns([1, 4])


with col1:
    st.image(Image.open("app/static/images/logo.png"), width=100)
with col2:
    st.title("SUSBot")
    st.write("A fullscreen live demo for chatbot consultation")


    
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="app/static/images/icons-bot.png").write(msg["content"])
    elif msg["role"] == "user":
        st.chat_message(msg["role"], avatar="app/static/images/tux.png").write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = openai.OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="app/static/images/tux.png").write(prompt)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview", 
        messages=st.session_state.messages,
        max_tokens=256,)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant", avatar="app/static/images/icons-bot.png").write(msg)