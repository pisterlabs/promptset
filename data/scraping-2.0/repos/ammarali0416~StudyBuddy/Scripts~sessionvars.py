# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sessionvars.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ammar syed ali <https://www.linkedin.co    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/11/19 13:57:12 by ammar syed        #+#    #+#              #
#    Updated: 2023/11/19 13:57:12 by ammar syed       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #
"""
Initialize session variables for the app
"""
from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from Scripts import azsqldb
from random import randint
import uuid


load_dotenv(find_dotenv())

def initialize_session_vars():
    '''
        LOGIN vars
    '''
    # Store the user information in a dictionary
    if "user_info" not in st.session_state:
        st.session_state.user_info = {'user_id': None,
                    'role': None,
                    'username': None}

    # Create a cursor object
    if "sqlcursor" not in st.session_state:
        st.session_state.sqlcursor = azsqldb.connect_to_azure_sql()

    '''
    Dashboard vars
    '''

    #####
    # Sidebar vars
    # Initialize the class information
    if "class_info" not in st.session_state:
        st.session_state.class_info = {'class_id': None,
                                    'class_name': None,
                                    'class_code': None,
                                    'index_name': None}
    # Store the class information
    if "class_info" not in st.session_state:
        st.session_state.class_info = None
    # Store the selected class so the dashboard remains the same after navigating to other pages
    if 'selected_class_name' not in st.session_state:
        st.session_state.selected_class_name = None
    # New class toggle (teacher)
    if 'show_new_class_input' not in st.session_state:
        st.session_state.show_new_class_input = False
    # Join class toggle (student)
    if 'show_join_class_input' not in st.session_state:
        st.session_state.show_join_class_input = False

    #####
    # Faqs vars
    # The FAQ toggle
    if 'show_faqs' not in st.session_state:
        st.session_state.show_faqs = False

    ####
    # upload file vars
    if 'show_upload_file' not in st.session_state:
        st.session_state.show_upload_file = False
    
    if 'show_upload_file2' not in st.session_state:
        st.session_state.show_upload_file2 = False
        
    # Initialize the upload counter in session state
    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = str(randint(0, 1000000))
    
        # Initialize the upload counter in session state
    if 'upload_key_2' not in st.session_state:
        st.session_state.upload_key_2 = str(randint(1000001, 10000000))

    ####
    # Module vars
        # Store the selected class so the dashboard remains the same after navigating to other pages
    if 'selected_module_name' not in st.session_state:
        st.session_state.selected_module_name = None
    # New module toggle (teacher)
    if 'new_module_toggle' not in st.session_state:
        st.session_state.new_module_toggle = False
    # Delete module toggle (teacher)
    if 'delete_module_toggle' not in st.session_state:
        st.session_state.delete_module_toggle = False
    # Store module information
    if "module_info" not in st.session_state:
        st.session_state.module_info = {
            'module_id': None,
            'module_name': None
        }
    # Store the selected modules for chatting
    if 'selected_modules' not in st.session_state:
        st.session_state.selected_modules = []  
    
    #### chat screen vars
    if 'context_selection_toggle' not in st.session_state:
        st.session_state.context_selection_toggle = True
    
    if 'blobs_df' not in st.session_state:
        st.session_state.blobs_df = None
    
    if 'blobs_to_retrieve' not in st.session_state:
        st.session_state.blobs_to_retrieve = None
    
    if 'ai_client' not in st.session_state:
        st.session_state.ai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY")
        )
    if "session_id" not in st.session_state: # Used to identify each session
        st.session_state.session_id = str(uuid.uuid4())

    if "run" not in st.session_state: # Stores the run state of the assistant
        st.session_state.run = {"status": None}

    if "messages" not in st.session_state: # Stores the messages of the assistant
        st.session_state.messages = []

    if "retry_error" not in st.session_state: # Used for error handling
        st.session_state.retry_error = 0
       
    if 'openai_fileids' not in st.session_state:
        st.session_state.openai_fileids = []
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if 'cleanup' not in st.session_state:
        st.session_state.cleanup = False
        
    if 'uploaded_to_openai' not in st.session_state:
        st.session_state.uploaded_to_openai = False