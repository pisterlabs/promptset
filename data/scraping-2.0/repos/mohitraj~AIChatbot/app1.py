import openai
import streamlit as st
from streamlit_chat import message

from key1 import KEY
from first1 import create_vector
# Setting page title and header
st.set_page_config(page_title="CodingWisdom", page_icon=":sparkles:")
st.markdown("<h1 style='text-align: center;'>Mohit Chatbot 🌞</h1>", unsafe_allow_html=True)

# Set org ID and API key
#openai.organization = "<YOUR_OPENAI_ORG_ID>"
openai.api_key = KEY

## Upload in side bar 

import streamlit as st
import os
import glob

saved_path = "data"
# Ensure the directory exists
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
  
# def remove_all_files(directory):
#    files = glob.glob(directory + '/*')
#    for f in files:
#        os.remove(f)

# Usage


# Upload the file
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

if uploaded_file is not None:
    # To read file as bytes:
    for file in uploaded_file:
        bytes_data = file.getvalue()
    
    # Save the uploaded file to the 'data' directory
        with open(os.path.join(saved_path, file.name), 'wb') as out_file:
            out_file.write(bytes_data)

    st.success('PDF file saved in data directory')
    create_vector()
    #remove_all_files(saved_path)
    st.success('Vector created')



# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=50)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        from second1 import generate_response
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        #st.session_state['model_name'].append(model_name)
 


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user1')
            message(st.session_state["generated"][i], key=str(i))
            
       
            
            
            
            
            