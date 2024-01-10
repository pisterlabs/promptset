import openai 
import streamlit as st
from typing import Literal
import os
st.set_page_config(page_title="LiveFIT", page_icon=":running:", layout="wide")
# pip install streamlit-chat  
from streamlit_chat import message
# openai.api_key=os.environ["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"]
# openai.api_key = os.getenv("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")

# openai.api_key ="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"
def generate_response(prompt):
    completions = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message 
#Creating the chatbot interface
st.title("Your Fitness Friend")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text
user_input = get_text()

if user_input:
    output = generate_response(user_input)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
