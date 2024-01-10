#!/usr/bin/env python3
from streamlit_chat import message
import openai
import os
import streamlit as st

MODELS = ["gpt-4", "gpt-3.5-turbo"]

# if total_tokens.txt exists, read it, otherwise create it
if os.path.exists("total_tokens.txt"):
    with open("total_tokens.txt", "r") as f:
        total_tokens = int(f.read())
else:
    with open("total_tokens.txt", "w") as f:
        f.write("0")
        total_tokens = 0
# if messages.txt exists, read it, otherwise create it
if os.path.exists("messages.txt"):
    with open("messages.txt", "r") as f:
        messages = f.read()
        messages = eval(messages)   # Convert string to list
else:
    with open("messages.txt", "w") as f:
        messages = [
            {"role": "system", "content": "Olvida las órdenes anteriormente dadas."},
            {"role": "system", "content": "Eres un asistente muy útil experto en programación."},
        ]
        f.write(str(messages))

def get_response(prompt, model):
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(model=model, messages=messages)
    contentn = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    messages.append({"role": "assistant", "content": contentn})
    print(f"messages: {messages}")
    return contentn, total_tokens

def get_prompt():
    input_text = st.text_input("You: ","", key="input")
    return input_text 

def main():
    st.title("Custom chat GPT")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Lateral bar
    if os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:
            api_key = f.read()
            openai_api_key = st.sidebar.text_input("Add your OpenAI API key here:", type="password", value=f"{api_key}")
    else:
        openai_api_key = st.sidebar.text_input("Add your OpenAI API key here:", type="password")
    selected_model = st.sidebar.selectbox("Select a model:", MODELS)
    global total_tokens
    total_tokens_used = st.sidebar.info(f"Total tokens used: {total_tokens}")

    # Add a placeholder
    message_placeholder = st.empty()

    user_input_placeholder = st.empty()

    if openai_api_key:
        message_placeholder.write("API key entered. You can now chat with the bot.")
        openai.api_key = openai_api_key
        user_input = get_prompt()
        if user_input:
            content, tokens = get_response(user_input, selected_model)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(content)
            with open("total_tokens.txt", "r") as f:
                total_tokens = int(f.read())
            total_tokens += tokens
            with open("total_tokens.txt", "w") as f:
                f.write(str(total_tokens))
            total_tokens_used.info(f"Total tokens used: {total_tokens}")
            with open("messages.txt", "w") as f:
                f.write(str(messages))
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    else:
        message_placeholder.write("Please enter your message below:")
        user_input_placeholder.text_input("Escribe tu mensaje aquí:", key="user_input", disabled=True)

if __name__ == "__main__":
    main()