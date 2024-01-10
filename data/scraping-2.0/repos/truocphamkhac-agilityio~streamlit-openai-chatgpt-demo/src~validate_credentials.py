import streamlit as st
from openai import OpenAI

# Validate OpenAI credentials
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

completion = client.chat.completions.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "What is Streamlit?"}]
)

st.write(completion.choices[0].message.content)
