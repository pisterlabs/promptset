import streamlit as st
from datetime import date, datetime
import pandas as pd
from io import StringIO
import json
import os

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from config import MODELS, TEMPERATURE, MAX_TOKENS, DATE_VAR, DATA_FRACTION, APP_NAME, MEAN_AGG, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR
from app_utils import (generate_responses, initialize_session_state, parse_pdf_document, create_knowledge_base, generate_kb_response)



def run_upload_and_settings():
    """This function runs the upload and settings container"""
    general_context = st.session_state["general_context"]
    brief_description = st.text_input("Please provide a brief description of the file (e.g. This is a research report on longevity)", "")
    if len(brief_description)>0:
            general_context = general_context + "The following brief description of the file was provided: "+ brief_description + "\n"
            st.session_state["general_context"] = general_context

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        #copy the file to "raw" folder
        with open(os.path.join("../data/raw/",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["uploaded_file"] = uploaded_file.name

        # read the file and process the pdf


def run_chatbot():
    template=""
    general_context = st.session_state["general_context"]
    model = st.session_state["generation_model"]

    # Start button
    start_button = st.button("Build Knowledge Base")

    if start_button:
        docs = parse_pdf_document(os.path.join("../data/raw/",st.session_state["uploaded_file"]))

        # process time series data to save to knowledge base
        create_knowledge_base(docs)
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input("What are the fields in my data?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):                
                response = generate_kb_response(prompt, model, template) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

        # if st.session_state["generated_responses"] and not st.session_state["cleared_responses"]:
        #     clear_button = st.button("Clear Responses")

        # if clear_button and not st.session_state["cleared_responses"]:        
        #     print(st.session_state["responses"])
        #     st.session_state["generated_responses"]=False
        #     st.session_state["responses"] = []
        #     st.session_state["cleared_responses"]=True

        # elif clear_button:
        #     st.write("No responses to clear - please generate responses")
        #         # responses = []
        #         # ratings = [None, None, None]

        #llm = HuggingFacePipeline(pipeline=pipeline)