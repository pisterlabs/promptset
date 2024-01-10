import openai
import streamlit as st


@st.cache_data
def create_content(transcript):

    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"Create a summary in markdown format for the given transcript\n{transcript}\n",
        temperature=0.7,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    markdown_content = response["choices"][0]["text"]
    return markdown_content
