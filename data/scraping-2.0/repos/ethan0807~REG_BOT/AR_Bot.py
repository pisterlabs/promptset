from LLM_Ranked_Retriever import get_retrieved_nodes
from LLM_Text_Summerizer import generate_summary_chat
from LLM_Index_Loader import load_index
import Globals
import openai
import streamlit as st
from streamlit_chat import message

# Streamlit UI for the Army RegBot
# Command line run: streamlit run AR_Bot.py --server.port 80

globals = Globals.Defaults()
openai.api_key = globals.open_api_key
model_name = globals.default_model
source = ""

index = load_index()

# Setting page title and header
st.set_page_config(page_title="REGBOT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Army RegBot</h1>",
            unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []

# Sidebar - let user clear the current conversation
st.sidebar.title("Enter a question about the following Army Regulations:")
st.sidebar.markdown(" AR735-5 Property Management, AR600-8-24 Officer Transfer and Discharge, AR 608-10 Child Development Services, AR750-1 Material Maintenance Management")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# generate a response


def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    nodes_with_score = get_retrieved_nodes(index, prompt, vector_top_k=1)
    if nodes_with_score[0].score < 0.75:
        response = "I'm sorry, I can't find a good answer to that question. Can you please rephrase it or attempt to be more specific?"
    else:
        summaries = generate_summary_chat(nodes_with_score)
        reg = nodes_with_score[0].node.metadata["regulation"]
        sec_num = nodes_with_score[0].node.metadata["section_number"]
        sec_name = nodes_with_score[0].node.metadata["section_name"]
        source = f"Source: Regulation {reg}, {sec_num} {sec_name}"
        summaries[0].response += "\n\n" + source
        response = summaries[0].response

    st.session_state['messages'].append(
        {"role": "assistant", "content": response})

    print(st.session_state['messages'])

    return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if 'generated' in st.session_state:
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
