import streamlit as st
from openai import OpenAI
import json
import sys
sys.path.append('C:/VS code projects/Road to Hack/auto_negotiator/Utilities/')
from request_functions import show_json
from request_functions import submit_message
from request_functions import get_response
from request_functions import pretty_print
from request_functions import wait_on_run
from request_functions import check_response
from request_functions import function_json
from request_functions import MATH_ASSISTANT_ID
from request_functions import thread
from request_functions import client
print(thread.id)
st.title("Request Bot")

requirements_path = 'C:/VS code projects/Road to Hack/auto_negotiator/Utilities/requirements.json'

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
    
    ## Send user message to assistant and display assistant response
    run = submit_message(client,MATH_ASSISTANT_ID, thread, prompt)
    run = wait_on_run(client,run, thread)
    completion = False
    if run.status == 'requires_action':
        response,completion = check_response(client,thread,run)
    else:
        response = pretty_print(get_response(client,thread))

    response = f"{response}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    
    if completion is True:
        print('done')
        json_data = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
        with open(requirements_path, 'w') as json_file:
            json.dump(json_data, json_file)
        st.json(json_data)
    else:
        print('not done')
    st.session_state.messages.append({"role": "assistant", "content": response})
