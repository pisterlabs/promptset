# %%writefile frontend.py
import requests
import streamlit as st
import utils
import openai
import os

api_key = "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"


# Replace with the URL of your backend
app_url = "http://127.0.0.1:8000/chat"



def openai_llm_response(user_input):
    """Send the user input to the LLM API and return the response."""
    # Append user question to the conversation history
    response_text = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=128,
        temperature=0.7,
        api_key=api_key
    ).choices[0].text

    return response_text


def main():
    st.title("🦸 A Plant Assistant Chatbot")

    col1, col2 = st.columns(2)
    with col1:
        utils.clear_conversation()
    # Get user input
    if user_input := st.text_input("Ask about Plants 👇", key="user_input", max_chars=100):
        res = openai_llm_response(user_input)
        st.write(res)
    # Display the cost
    st.caption(f"Total cost of this session: US${st.session_state.total_cost}")

    # Display the entire conversation on the frontend
    utils.display_conversation(st.session_state.conversation_history)

    # Download conversation code runs last to ensure the latest messages are captured
    with col2:
        utils.download_conversation()


if __name__ == "__main__":
    main()
