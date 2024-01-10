import streamlit as st
from openai import OpenAI
import json
import sys
sys.path.append('C:/VS code projects/Road to Hack/auto_negotiator/Utilities/')
from negotiation_utils import *
from negotiation_utils import gpt_draft_mail
from negotiation_utils import submit_message
from negotiation_utils import get_response
from negotiation_utils import pretty_print
from negotiation_utils import wait_on_run
from negotiation_utils import check_response
from negotiation_utils import MATH_ASSISTANT_ID
from negotiation_utils import thread
from negotiation_utils import client
print(thread.id)
st.title("Procurement Bot")
# Path to requirements file
requirements_path = 'C:/VS code projects/Road to Hack/auto_negotiator/Utilities/procurement.json'
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display initial response
initialsiation = True
if initialsiation:
    specification = json.loads(json.load(open('C:/VS code projects/Road to Hack/auto_negotiator/Utilities/requirements.json','r')))
    response = gpt_draft_mail(specification, 'ABC Company')
    st.session_state.initialised = True

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Send user message to assistant and display assistant response
    run = submit_message(client, MATH_ASSISTANT_ID, thread, prompt)
    run = wait_on_run(client, run, thread)
    completion = False
    if run.status == 'requires_action':
        response,completion = check_response(client,thread,run)
    else:
        response = pretty_print(get_response(client,thread))
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    if completion is True:
        print('done')
        print(run.status)
        json_data = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
        with open(requirements_path, 'w') as json_file:
            json.dump(json_data, json_file)
        st.json(json_data)
    else:
        print('not done')
        print(run.status)
    st.session_state.messages.append({"role": "assistant", "content": response})
