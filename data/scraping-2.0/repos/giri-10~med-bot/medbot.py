import openai
from openai import OpenAI
import streamlit as st
import os
from keywords import medical_keywords

st.set_page_config(page_title="MedBot - Medical Assistant")
st.title("MedBot")

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def is_medical_question(user_input):
    """
    Determines if the given user input is a medical question.

    Args:
        user_input (str): The user's input.

    Returns:
        bool: True if the user input is a medical question, False otherwise.
    """

    # Convert user_input to lowercase and split it into words
    words = user_input.lower().split()

    # Check if any of the words in user_input are in medical_keywords
    for word in words:
        if word in medical_keywords:
            return True

    # If none of the words in user_input are in medical_keywords, return False
    return False

try:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        message_placeholder = st.empty() 
        bot_response = ""

        if not is_medical_question(prompt):
            bot_response = "I'm sorry, I can only provide information about medical topics."
        else:
            with st.chat_message("assistant"):
                for response in client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                ):
                    bot_response += (response.choices[0].delta.content or "")
                    # bot_response += (response.choices[0].text or "")
            message_placeholder.markdown(bot_response + "▌")


        message_placeholder.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
except openai.Error as e:
    st.error(f"An error occurred with the OpenAI API: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

    