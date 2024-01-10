import streamlit as st
import langchain_helper as lch
from dotenv import load_dotenv
import os

load_dotenv("openapi_key.txt")
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("🤖 Dataroots website chatbot")

# Create a list to store the entered URLs persistently throughout the session
if "url_list" not in st.session_state:
    st.session_state.url_list = []

# Create a dictionary to track checked/unchecked status of URLs
if "url_status" not in st.session_state:
    st.session_state.url_status = {}


# Function to display entered URLs with checkboxes
def display_urls_with_checkbox(urls):
    st.sidebar.header("Entered URLs")
    for url in urls:
        checkbox_state = st.sidebar.checkbox(url, key=url)
        st.session_state.url_status[url] = checkbox_state


display_urls_with_checkbox(st.session_state.url_list)


def submit():
    new_url = st.session_state.new_url
    if new_url not in st.session_state.url_list:
        st.session_state.url_list.append(new_url)
        st.session_state.url_status[new_url] = False
        st.session_state.url_list.sort()
        st.session_state.new_url = ""


# Text input for adding a new URL
st.text_input(label="URL", key="new_url", on_change=submit)

# Consolidate knowledge button
if st.button("Consolidate knowledge"):
    selected_urls = [
        url for url, status in st.session_state.url_status.items() if status
    ]
    if len(selected_urls) == 0:
        st.sidebar.warning("Please select at least one URL!")
    else:
        st.sidebar.success("Created vector store")
        st.session_state.vector_store = lch.create_vector_store(selected_urls)

# Temperature slider
temperature = st.sidebar.slider(
    "Select temperature", min_value=0.0, max_value=1.0, value=0.0
)

# User input for asking a question
user_input = st.text_area("Ask your question")

# Generate button
if st.button("Submit question"):
    if st.session_state.vector_store and user_input:
        response = lch.ask_dataroots_chatbot(
            openai_api_key=openai_api_key,
            temperature=temperature,
            vector_store=st.session_state.vector_store,
            user_input=user_input,
        )
        st.text_area(
            label="Answer",
            value=response["answer"],
            height=200,
        )
        st.markdown(body=f'Sources: {response["sources"]}', unsafe_allow_html=True)
