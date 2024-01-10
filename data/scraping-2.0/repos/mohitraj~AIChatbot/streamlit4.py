import os
import streamlit as st
import shutil
import openai
import streamlit as st
from streamlit_chat import message
from second4 import generate_response

from key1 import KEY
from first4 import create_vector
# Setting page title and header
st.set_page_config(page_title="Mohit", page_icon=":sparkles:")
st.markdown("<h1 style='text-align: center;'>Mohit Chatbot 🌞</h1>", unsafe_allow_html=True)

# Set org ID and API key
#openai.organization = "<YOUR_OPENAI_ORG_ID>"
openai.api_key = KEY

## Upload in side bar
flag = 0
import streamlit as st
import os
import glob

saved_path = "data"
# Ensure the directory exists
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

saved_path = "data_dir"
# This is your custom function
def custom_function():
    st.write(f"Hello EveryONE")

# Get the subdirectories in the 'data' directory
subdirs = [name for name in os.listdir(f'./{saved_path}') if os.path.isdir(os.path.join(f'./{saved_path}', name))]

selected_dir = st.sidebar.radio('Select a directory:', options=subdirs)

# Create radio buttons in the sidebar for each subdirectory


# Create a file uploader for PDF files
uploaded_file = st.sidebar.file_uploader('Upload a PDF', type=['pdf'])

# Add a submit button in the sidebar
st.write("Selected Dir",selected_dir)
if uploaded_file is not None:
    # Get the path of the selected subdirectory
    save_path = os.path.join(f'./{saved_path}', selected_dir)
    # Save the uploaded file in the selected subdirectory
    with open(os.path.join(save_path, uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

        # Run the custom function


        st.success('File uploaded successfully.')

st.sidebar.write("Create Vectors")
if st.sidebar.button('Submit'):
    flag = 1

    create_vector(selected_dir)
    
st.sidebar.write("Select Folder for Query ")
checkbox_states = {}
for folder_name in subdirs:
    checkbox_states[folder_name] = st.sidebar.checkbox(folder_name)

# Display the selected checkboxes

selected_folders = [folder_name for folder_name, state in checkbox_states.items() if state]
#st.write("Selected Folders:", selected_folders)




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
        #from second1 import generate_response
        output = generate_response(user_input,selected_folders)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        #st.session_state['model_name'].append(model_name)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user1')
            message(st.session_state["generated"][i], key=str(i))
