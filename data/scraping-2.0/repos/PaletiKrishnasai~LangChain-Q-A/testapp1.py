import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the chat model
chat = ChatOpenAI(temperature=0.5)

# Streamlit UI setup
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

# Initialize session state for storing messages
if "flowmessages" not in st.session_state:
    st.session_state["flowmessages"] = [
        SystemMessage(content="test content.")
    ]


# Function to get model response
def get_chatmodel_response(question):
    st.session_state["flowmessages"].append(HumanMessage(content=question))
    answer = chat(st.session_state["flowmessages"])
    st.session_state["flowmessages"].append(AIMessage(content=answer.content))
    return answer.content


# User input
input_question = st.text_input("Input: ", key="input")

# Button to submit the question
submit = st.button("Ask the question")

# Display response when button is clicked
if submit and input_question.strip():
    response = get_chatmodel_response(input_question)
    st.subheader("The Response is")
    st.write(response)
else:
    st.write("Please enter a question and press 'Ask the question'.")

# Optional: Button to reset conversation
if st.button("Reset Conversation"):
    st.session_state["flowmessages"] = []
    st.experimental_rerun()
