import streamlit as st
import openai

st.set_page_config(page_title="OpenAI Text Gen", page_icon="🤖")

openai.api_key = st.secrets["openai_api_key"]

st.title("AI Text Gen 🤖")
st.subheader("Write the Prompt below")
prompt = st.text_area("Prompt", "Enter the Prompt Here...")

ask_button = st.button("Generate")
if ask_button:
    st.subheader("OpenAI Davinci-3 AI")
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{prompt}",
        max_tokens=100,
        temperature=0
    )
    text = completion.choices[0].text
    st.write(text)
