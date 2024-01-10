# coding=utf-8
import openai
import yaml
import pathlib
import streamlit as st
base_path = pathlib.Path.cwd().parent.parent

from chatgpt_35.utils import get_config
from chatgpt_35.utils.openai_wrapper import ChatGPT

config_file = base_path / 'config' / 'config.yaml'
config = get_config.run(config_file)


def main():
    st.title("ChatGPT Demo using Streamlit")

    # Initialize conversation_history in session_state if not exists
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    input_text = st.text_input("Enter your message:")
    if st.button("Send"):
        # Add user's message to conversation history
        st.session_state.conversation_history.append({"sender": "User", "message": input_text})

        prompt = f"User: {input_text}\nAI:"

        chatgpt = ChatGPT(config.openai_api_key, config.openai_model)
        prompt = prompt.format(**{'input_text':input_text})
        gpt_response = chatgpt.run_chat(prompt)

        # Add AI's response to conversation history
        st.session_state.conversation_history.append({"sender": "AI", "message": gpt_response})

        # Keep only the last 10 messages
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]

    # Display conversation history
    st.write("Conversation history:")
    for msg in st.session_state.conversation_history:
        st.write(f"{msg['sender']}: {msg['message']}")

if __name__ == '__main__':
    main()