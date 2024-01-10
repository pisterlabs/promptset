import openai
import streamlit as st
from random import choice


# pip install streamlit-chat
from streamlit_chat import message


#creating a function which will generate the calls from the api

def generate_response(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt = prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,

    )
    message = completions.choices[0].text
    return message


# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

def show_chat_page():

    st.title("chatBot: Streamlit+OpenAI")

    openai.api_key = st.secrets["api_secret"]

    user_input = get_text()


    if user_input:
        output=generate_response(user_input)
        # store the output
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


