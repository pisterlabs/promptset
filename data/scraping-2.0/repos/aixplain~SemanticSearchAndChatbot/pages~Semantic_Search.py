import streamlit as st
from ingestor_api import query
from utils import parse_search_response
import os
import openai
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
load_dotenv(os.path.join(BASE_DIR, "secrets.env"), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG DEMO")

st.title("_RAG DEMO_")
st.subheader("Internal documents search and QA")

if "files" not in st.session_state:
    st.session_state["files"] = []

if "responses" not in st.session_state:
    st.session_state["responses"] = []

if "collection_name" not in st.session_state:
    st.session_state["collection_name"]="Combined Documents"



st.subheader("Question Answering")
with st.form(key='Search Document',clear_on_submit=True):
    question = st.text_input("Enter your Query", placeholder="Type in your question...", label_visibility="collapsed")      
    limit = st.number_input("Number of relevant results",value=10)
    
    SearchButton = st.form_submit_button(label = 'Search')


if SearchButton:
    st.write(f"**Question:** {question}")
    response_search = query(question, num_results=limit)
    st.session_state["files"], summary = parse_search_response(response_search)
    st.session_state["responses"].append(response_search)

    st.write("Documents used for answering:")
    for i, file in enumerate(st.session_state["files"]):
        with st.expander(f"**Search result number {i+1}:**"):
            st.write(f"file: [{file['file_name']}]({file['file_url']})")
            # st.write(f"file_name: {file['file_name']}")
            st.write(f"{file['text']}")

    # response = answer_query(st.session_state["collection_name"], question)
    # answer=response.json()["message"]
    st.write(f"**Summary of the top results:** {summary}")
    st.write("Done!")
    

    st.write(f"**Answer:**")
    chunks = ""
    content_chunks=5
    for i, chunk in enumerate(st.session_state["files"][:content_chunks]):
        chunks += f"Document {i+1}:\r\n\r\nfile_name:\r\n\r\n" + chunk["file_name"] + "\r\n\r\n" + "file_url:\r\n\r\n" + chunk["file_url"] + chunk["text"] + "\r\n\r\n"

    search_prompts = f"""
        Question: {question}
        Use only the following document chunks to answer the question:
        **{chunks}**
        Only if any of the documents above was used in your response, specify the file_name and file_url (if exists) at the end of your response as reference in seprate lines with following format:
        Reference:\n
        file_name (file_url)
        if none of the documnets were used do not include the reference.
        do not show repeated reference if they are the same.
        Answer:
        """
    st.session_state.messages.append({"role": "system", "content": search_prompts})

    response = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.write(msg)