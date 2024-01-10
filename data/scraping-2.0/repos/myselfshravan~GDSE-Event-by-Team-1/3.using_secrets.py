import streamlit as st
import openai

st.set_page_config(page_title="My GPT", page_icon="🤖")

openai.api_key = st.secrets["openai_api_key"]

st.title("AI Text Gen 🤖")
prompt = st.text_area("Prompt", "Enter the prompt Here...")

ask_button = st.button("Generate")
if ask_button:
    st.subheader("OpenAI Davinci-3 AI")
    completion = openai.Completion.create(
        model="text-davinci-001",
        prompt=f"{prompt}",
        max_tokens=100,
        temperature=0
    )
    text = completion.choices[0].text
    st.write(text)
