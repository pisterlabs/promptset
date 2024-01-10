import streamlit as st
import openai
import requests

# Set your GitHub API token here
GITHUB_TOKEN = "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"

# Set your OpenAI API key here
openai.api_key = "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"

# Streamlit app title
st.title("GitHub Code Search and Chat App")

# User input for GitHub repository and code search query
repo_name = st.text_input("Enter the GitHub repository name (e.g., username/repo):")
user_query = st.text_input("Enter your code search query:")

# GitHub API call to get code snippets
if repo_name and user_query:
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    url = f"https://api.github.com/search/code?q={user_query}+repo:{repo_name}"
    response = requests.get(url, headers=headers)
    data = response.json()

    # Display code snippets
    if "items" in data:
        st.header("Code Search Results:")
        for item in data["items"]:
            st.code(item)

# Chat section for refining the query
st.sidebar.header("Chat with the Model")

# User input for chat
user_chat_input = st.sidebar.text_input("You: ")

# OpenAI API call for chat-based interaction
if user_chat_input:
    chat_history = f"Search for code related to: {user_query}\nYou: {user_chat_input}\n"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=chat_history,
        max_tokens=150,
        n=1,
    )
    model_reply = response['choices'][0]['text']

    # Display model's reply
    st.sidebar.text("Model: " + model_reply)

# Note: Replace "YOUR_GITHUB_TOKEN" and "YOUR_OPENAI_API_KEY" with your actual GitHub token and OpenAI API key
