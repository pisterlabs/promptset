import openai
import streamlit as st
from streamlit_chat import message
import os
import toml

# load environment variables
# toml_dict = toml.load('SECRETS.toml')
# OPENAI_API_KEY = toml_dict['openai_api_key']
OPENAI_API_KEY = st.secrets["openai_api_key"] 

openai.api_key = OPENAI_API_KEY

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.Completion.create(
        engine="text-davinci-002",  # Use "text-davinci-003" for ChatGPT
        max_tokens=50,
        prompt=prompt
    )
    print(completion)
    response = completion.choices[0].text.strip()
    st.session_state['messages'].append({"role": "bot", "content": response})

def main():
    generated_pointer = len(st.session_state['messages'])
    st.title("GPTeacher")

    # chat container
    chat_container = st.container()

    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("You:", key='input')
            submit_button = st.form_submit_button(label='Send')

            if user_input and submit_button:
                generate_response(user_input)

    if len(st.session_state['messages']) != generated_pointer:
        generated_pointer = len(st.session_state['messages'])
        with chat_container:
            for msg in st.session_state['messages']:
                if msg['role'] == 'user':
                    message(msg['content'], is_user=True)
                else:
                    message(msg['content'], is_user=False)


if __name__ == "__main__":
    main()
