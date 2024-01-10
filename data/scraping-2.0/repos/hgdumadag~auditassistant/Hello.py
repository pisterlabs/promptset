# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import streamlit as st
from streamlit.logger import get_logger
from openai import OpenAI
import time
import os
#import json
#from dotenv import load_dotenv
#load_dotenv()

LOGGER = get_logger(__name__)



# Set your OpenAI Assistant ID here
assistant_id = ""
os.environ["assistant_id"] = st.secrets["ASSISTANT_ID"]
assistant_id=st.secrets.ASSISTANT_ID
#assistant_id = os.getenv("ASSISTANT_ID")

# Initialize the OpenAI client (ensure to set your API key in the sidebar within the app)
client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# Initialize the session state variables for file IDs and chat control
if "file_id_list" not in st.session_state:
    st.session_state["file_id_list"] = []

if "start_chat" not in st.session_state:
    st.session_state["start_chat"] = False

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="JGSHI Audit Assistant", page_icon=":robot_face:", layout="wide")
st.header(":robot_face: JGSHI Audit Assistant")

# Get the OPENAI API Key
#openai_api_key_env = os.getenv("OPENAI_API_KEY")
openai_api_key_env=""
os.environ["openai_api_key_env"] = st.secrets["OPENAI_API_KEY"]
openai_api_key = st.sidebar.text_input(
    'OpenAI API Key', placeholder='sk-', value=openai_api_key_env)
openai_api_key = openai_api_key_env
url = "https://platform.openai.com/account/api-keys"
st.sidebar.markdown(f"Get your API key [here]({url})")
if openai_api_key:
    OpenAI.api_key = openai_api_key

# Button to start the chat session
if st.sidebar.button("Start Chat"):
    st.session_state["start_chat"] = True
    # Create a thread once and store its ID in the session state
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    st.write("thread id: ", thread.id)

# Define the function to process messages with citations
def process_message_with_citations(message):
    message_content = message.content[0].text.value
    return message_content

# Only show the chat interface if the chat has been started
if st.session_state.start_chat:
    # st.write(getStockPrice("AAPL"))
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for the user
    if prompt := st.chat_input("How can I help you?"):
        #Add user message to the state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Add the user's message to the existing thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt
        )

        # Create a run with additional instructions
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=assistant_id,
            instructions="As an Expert Internal Auditor Assistant, you possess extensive knowledge and practical experience in various audit processes, including financial, operational, compliance, and information technology audits. Your expertise encompasses a deep understanding of audit standards, risk management, and control processes. You are adept at identifying potential risks, inefficiencies, and areas for improvement within an organization's operations. You are an expert in the nternational Professional Practices Framework."
        )

        # Poll for the run to complete and retrieve the assistant's messages
        while run.status not in ["completed" , "failed"]:
            st.sidebar.write(run.status)
            # if run.status == "requires action":
            #    handle_function(run)
            time.sleep(2)	
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
            st.sidebar.write(run.status)
        
        if run.status == "failed":
            # Log the error message
            st.error(f"Run failed with error: {run.error_message}")
            # You can take additional actions here based on the error

        # Retrieve messages added by the assistant
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        )

        # Process and display assistant's messages
        assistant_messages_for_run = [
            message for message in messages
            if message.run_id == run.id and message.role == "assistant"
        ]
        for message in assistant_messages_for_run:
            full_response = process_message_with_citations(message)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response, unsafe_allow_html=True)
#######
#def run():
#    st.set_page_config(
#        page_title="Welcome to the Audit Assistant Page",
#        page_icon="👋",
#    )
# 
#if __name__ == "__main__":
#    run()
#####