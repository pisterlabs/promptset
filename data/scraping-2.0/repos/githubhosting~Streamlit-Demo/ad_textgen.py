import streamlit as st
import openai


st.set_page_config(page_title="My GPT - by Shravan", page_icon="🤖")

openai.api_key = "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"

st.title("AI Text Gen 🤖")
st.subheader("Write the Prompt")
prompt = st.text_area("Prompt", "Enter the Text Here")

check = st.button("Ask the AI")
if check:

    st.subheader("OpenAI Davinci-3 AI")
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{prompt}",
        max_tokens=100,
        temperature=0
    )
    text = completion.choices[0].text
    st.write(text)
