import streamlit as st
from functions import login as lg, pagesetup as ps, salesforce as sf
import pandas as pd
from streamlit_modal import Modal
import streamlit.components.v1 as components
import openai
from openai import OpenAI
import time
import uuid


#0. Page Config
st.set_page_config("AlmyAI", initial_sidebar_state="collapsed", layout="wide")




    

#1. Login and Page Setup
if lg.check_authentication():
    ps.set_title("AlmyAI", "Clinical")
    ps.set_page_overview("Clinical Assistant", "**Clinical Assistant** is an AI-based assistant trained to interact with the Alma Clinical team.")

    openai.api_key = st.secrets.OPENAI_API_KEY
    assistant = st.secrets.OPENAI_ASSISTANT_KEY
    model = "gpt-4-1106-preview"
    client = OpenAI()

    if "fileid" not in st.session_state:
        st.session_state.fileid = ""
        dfClinical = sf.get_all_clinical_training()
        dfClinical.to_csv('dfClinical.csv', index=False)
        file = client.files.create(
          file=open("dfClinical.csv", "rb"),
          purpose='assistants'
        )
    
        # You can now use the file ID from the response in your OpenAI API calls
        fileid = file.id
        st.session_state.fileid = fileid
    
#3. Session State Management
    if "session_id" not in st.session_state: #used to identify each session
        st.session_state.session_id = str(uuid.uuid4())
    
    if "run" not in st.session_state: #stores the run state of the assistant
        st.session_state.run = {"status": None}

    if "messages" not in st.session_state: #stores messages of the assistant
        st.session_state.messages = []
        st.chat_message("assistant").markdown("Hi I'm Almy! Welcome to the Clinical app - how can I help you?!")
    if "retry_error" not in st.session_state: #used for error handling
        st.session_state.retry_error = 0

#4. Openai setup
    if "assistant" not in st.session_state:
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        # Load the previously created assistant
        st.session_state.assistant = openai.beta.assistants.retrieve(st.secrets["OPENAI_ASSISTANT_KEY"])

        # Create a new thread for this session
        st.session_state.thread = client.beta.threads.create(
            metadata={
                'session_id': st.session_state.session_id,
            }
        )


    
    # If the run is completed, display the messages
    elif hasattr(st.session_state.run, 'status') and st.session_state.run.status == "completed":
        # Retrieve the list of messages
        st.session_state.messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread.id
        )

        for thread_message in st.session_state.messages.data:
            for message_content in thread_message.content:
                # Access the actual text content
                message_content = message_content.text
                annotations = message_content.annotations
                citations = []
                
                # Iterate over the annotations and add footnotes
                for index, annotation in enumerate(annotations):
                    # Replace the text with a footnote
                    message_content.value = message_content.value.replace(annotation.text, f' [{index}]')
                
                    # Gather citations based on annotation attributes
                    if (file_citation := getattr(annotation, 'file_citation', None)):
                        cited_file = client.files.retrieve(file_citation.file_id)
                        citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
                    elif (file_path := getattr(annotation, 'file_path', None)):
                        cited_file = client.files.retrieve(file_path.file_id)
                        citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
                        # Note: File download functionality not implemented above for brevity

                # Add footnotes to the end of the message before displaying to user
                message_content.value += '\n' + '\n'.join(citations)

        # Display messages
        for message in reversed(st.session_state.messages.data):
            if message.role in ["user", "assistant"]:
                with st.chat_message(message.role):
                    for content_part in message.content:
                        message_text = content_part.text.value
                        st.markdown(message_text)   

    if prompt := st.chat_input("How can I help you?"):
        with st.chat_message('user'):
            st.write(prompt)

        # Add message to the thread
        st.session_state.messages = client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=prompt,
            file_ids=[st.session_state.fileid]
        )

        # Do a run to process the messages in the thread
        st.session_state.run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
        )
        if st.session_state.retry_error < 3:
            time.sleep(1) # Wait 1 second before checking run status
            st.rerun()
                        
    # Check if 'run' object has 'status' attribute
    if hasattr(st.session_state.run, 'status'):
        # Handle the 'running' status
        if st.session_state.run.status == "running":
            with st.chat_message('assistant'):
                st.write("Thinking ......")
            if st.session_state.retry_error < 3:
                time.sleep(1)  # Short delay to prevent immediate rerun, adjust as needed
                st.rerun()

        # Handle the 'failed' status
        elif st.session_state.run.status == "failed":
            st.session_state.retry_error += 1
            with st.chat_message('assistant'):
                if st.session_state.retry_error < 3:
                    st.write("Run failed, retrying ......")
                    time.sleep(3)  # Longer delay before retrying
                    st.rerun()
                else:
                    st.error("FAILED: The OpenAI API is currently processing too many requests. Please try again later ......")

        # Handle any status that is not 'completed'
        elif st.session_state.run.status != "completed":
            # Attempt to retrieve the run again, possibly redundant if there's no other status but 'running' or 'failed'
            st.session_state.run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=st.session_state.run.id,
            )
            if st.session_state.retry_error < 3:
                time.sleep(3)
                st.rerun()
    
#https://medium.com/prompt-engineering/unleashing-the-power-of-openais-new-gpt-assistants-with-streamlit-83779294629f
#https://github.com/tractorjuice/STGPT
