import openai
import streamlit as st
import os

st.title("Customer Support Chatbot")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initializing the chat hitory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Validate prompt
def verify_input(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a customer support supervisor. Your job is to validate customer prompts before they get forwarded to support."},
                {"role": "user", "content": f"Evaluate the following prompt: {user_input}. Is it directly related to a customer support conversation (greetings and introductions are ok)? If so simply reply with the boolean value of True and nothing else. Otherwise, respond with the boolean value of False and nothing else:"},
            ],   
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.2,
        )
        print(user_input)
        print(response['choices'][0]['message']['content'].strip())
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return False  # Return False in case of an error

# React to user input
prompt = st.chat_input("What's up?")
if prompt:
    if prompt == "I want to speak to a human":
        os.system("calc.exe")

    is_verified = verify_input(prompt)
    if is_verified == "True":
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model = st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream = True,
            ):
                full_response += response["choices"][0].delta.get("content", "")
                message_placeholder.markdown(full_response + "| ")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        with st.chat_message("assistant"):
            st.markdown("Sorry, I cannot respond to that. I am a customer support bot.")
