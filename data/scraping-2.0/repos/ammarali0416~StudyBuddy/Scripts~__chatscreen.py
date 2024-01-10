# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    __chatscreen.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ammar syed ali <https://www.linkedin.co    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/02 15:17:52 by ammar syed        #+#    #+#              #
#    Updated: 2023/12/02 15:17:52 by ammar syed       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #
import streamlit as st
import os
import openai
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from Scripts import azsqldb, sessionvars, azblob as azb
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

sessionvars.initialize_session_vars()

def delete_files_from_openai():
    total_files = 0
    progress_value = 0

    # First list of files
    files = st.session_state.ai_client.files.list()
    total_files += len(files.data) if files.data else 0

    # Second list of files
    assistant_files = st.session_state.ai_client.beta.assistants.files.list(
        assistant_id=os.getenv("OPENAI_ASSISTANT")
    )
    total_files += len(assistant_files.data) if assistant_files.data else 0

    # Create a progress bar
    progress_bar = st.progress(0, text='Getting ready to study :)!')

    # Function to update progress
    def update_progress():
        nonlocal progress_value
        progress_value += 1
        progress_bar.progress(progress_value / total_files)

    # Delete files from the first list
    for file in files.data:
        file_id = file.id
        st.session_state.ai_client.files.delete(file_id=file_id)
        print(f"Deleted file {file_id}")
        update_progress()

    # Delete files from the second list
    for file in assistant_files.data:
        file_id = file.id
        st.session_state.ai_client.beta.assistants.files.delete(
            assistant_id=os.getenv("OPENAI_ASSISTANT"),
            file_id=file_id
        )
        print(f"Deleted file {file_id}")
        update_progress()

    # Complete the progress bar if there were no files
    if total_files == 0:
        progress_bar.progress(1)
        progress_bar.empty()



def context_selection():
    """
    A widget that allows the user to select what context the chatbot has access to
    """
    modules = azsqldb.get_modules(st.session_state.class_info['class_id'], st.session_state.sqlcursor)
    context_container = st.container()
    with context_container:
        st.info("Choose the modules StudyBuddy will help you with")
        # Counter to track the current column
        col_counter = 0
        # Iterating over the modules
        for module_name, module_id in modules.items():
            # Every three modules, create a new row of columns
            if col_counter % 3 == 0:
                col1, col2, col3 = st.columns(3)

            # Place the checkbox in the current column
            if col_counter % 3 == 0:
                with col1:
                    st.checkbox(module_name, key=f"module_{module_id}")
            elif col_counter % 3 == 1:
                with col2:
                    st.checkbox(module_name, key=f"module_{module_id}")
            elif col_counter % 3 == 2:
                with col3:
                    st.checkbox(module_name, key=f"module_{module_id}")

            # Increment the column counter
            col_counter += 1
        
        # Add a button to submit the selected modules
        if st.button("Let's Study!"):
            # Get the selected modules
            selected_modules = []
            for module_name, module_id in modules.items():
                if st.session_state[f"module_{module_id}"]:
                    selected_modules.append(module_name)
            # If the user didn't select any modules, display a warning
            if len(selected_modules) == 0:
                st.warning("Please select at least one module")
            else:
                st.session_state.selected_modules = selected_modules
                st.session_state.context_selection_toggle = False
                st.experimental_rerun()

def initialize_chat():
    module_learning_outcomes, class_learning_outcomes = azsqldb.get_learning_outcomes(
                                                            st.session_state.class_info['class_id'], 
                                                            st.session_state.selected_modules, 
                                                            st.session_state.sqlcursor)

    faq_df = azsqldb.get_questions_usernames(st.session_state.class_info['class_id'], st.session_state.sqlcursor) 

    initial_prompt = f"""
<INFO> INITIAL PROMPT </INFO>
You're chatting with {st.session_state.user_info['username']}\n
Their role is : {st.session_state.user_info['role']}\n
The class is : {st.session_state.class_info['class_name']}\n
The class learning outcomes are:\n {class_learning_outcomes}\n
You are going to discuss the following modules:\n
"""

    for module, outcome in module_learning_outcomes.items():
        initial_prompt += f" -Module: {module}\n\n"
        initial_prompt += f" -Learning outcomes: {outcome}\n\n"

    initial_prompt += f"Here is info on the files you recieved:\n\n{st.session_state.blobs_to_retrieve} \n\n"

    initial_prompt += f"Here are the FAQs for this class:\n\n{faq_df}"

    return initial_prompt


def upload_files_ai(blob_paths):
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    container_client = blob_service_client.get_container_client(os.getenv("AZURE_CONTAINER"))
    client = st.session_state.ai_client

    # Base directory (assuming this script is in the /scripts subdirectory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths
    staging_dir = os.path.join(base_dir, 'staging')
    json_path = os.path.join(base_dir, '.bin', 'files.json')

    # Ensure directories exist
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # List to store file objects
    uploaded_files = []

    for blob_path in blob_paths:
        # Adjust the path to save in the staging directory
        staging_path = os.path.join(staging_dir, os.path.basename(blob_path))

        # Download the file from Azure Blob
        blob_client = container_client.get_blob_client(blob_path)
        with open(staging_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        # Upload the file to OpenAI
        with open(staging_path, "rb") as file:
            response = client.files.create(file=file, purpose="assistants")
            uploaded_files.append(response)

        # Delete the file from the staging directory
        os.remove(staging_path)
    
    # Return this list of file ids
    return [file_obj.id for file_obj in uploaded_files]